# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
import ddn_utils
from ddn_utils import *
import boxx
import sddn

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
import training.dataset

# ----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).


def ddn_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    *args,
    **kwargs,
):
    d = {"batch_size": len(latents)}
    if "batch_seeds" in kwargs:
        total_output_level = kwargs.get("total_output_level", 2000)
        d["batch_seeds"] = d["idx_gens"] = kwargs["batch_seeds"]
        if "sampler" in kwargs:
            d["sampler"] = kwargs["sampler"]
        else:
            if "markov_sampler" in kwargs:
                d["idx_ks"] = torch.cat(
                    [
                        kwargs["markov_sampler"].sample(seed=seed)[:, None]
                        for seed in kwargs["batch_seeds"].tolist()
                    ],
                    -1,
                )  # l, b
            else:
                d["idx_ks"] = torch.cat(
                    [
                        torch.rand(
                            total_output_level,
                            1,
                            generator=torch.Generator().manual_seed(seed),
                        )
                        for seed in kwargs["batch_seeds"].tolist()
                    ],
                    -1,
                )  # l, b
    with torch.no_grad():
        d = net(d, None, class_labels)
    kwargs["total_output_level"] = d.get("output_level", -2) + 1
    boxx.mg()
    if boxx.cf.debug:
        showd(d, 1)
    return d


def edm_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# ----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.


def ablation_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=None,
    sigma_max=None,
    rho=7,
    solver="heun",
    discretization="edm",
    schedule="linear",
    scaling="none",
    epsilon_s=1e-3,
    C_1=0.001,
    C_2=0.008,
    M=1000,
    alpha=1,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    assert solver in ["euler", "heun"]
    assert discretization in ["vp", "ve", "iddpm", "edm"]
    assert schedule in ["vp", "ve", "linear"]
    assert scaling in ["vp", "none"]

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = (
        lambda beta_d, beta_min: lambda t: (
            np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
        )
        ** 0.5
    )
    vp_sigma_deriv = (
        lambda beta_d, beta_min: lambda t: 0.5
        * (beta_min + beta_d * t)
        * (sigma(t) + 1 / sigma(t))
    )
    vp_sigma_inv = (
        lambda beta_d, beta_min: lambda sigma: (
            (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
        )
        / beta_d
    )
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma**2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
            discretization
        ]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = (
        2
        * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == "vp":
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == "ve":
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == "iddpm":
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1
            ).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[
            ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
            .round()
            .to(torch.int64)
        ]
    else:
        assert discretization == "edm"
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == "linear"
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == "vp":
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == "none"
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= sigma(t_cur) <= S_max
            else 0
        )
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (
            sigma(t_hat) ** 2 - sigma(t_cur) ** 2
        ).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (
            sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
        ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == "euler" or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == "heun"
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(
                torch.float64
            )
            d_prime = (
                sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)
            ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * (
                (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
            )

    return x_next


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


@click.command()
@click.option(
    "--network",
    "network_pkl",
    help="Network pickle filename",
    metavar="PATH|URL",
    type=str,
    required=True,
)
@click.option(
    "--outdir",
    help="Where to save the output images",
    metavar="DIR",
    type=str,
    # required=True,
)
@click.option(
    "--seeds",
    help="Random seeds (e.g. 1,2,5-10)",
    metavar="LIST",
    type=parse_int_list,
    default="0-99",
    show_default=True,
)
@click.option(
    "--subdirs", help="Create subdirectory for every 1000 seeds", is_flag=True
)
@click.option(
    "--class",
    "class_idx",
    help="Class label  [default: random]",
    metavar="INT",
    type=click.IntRange(min=0),
    default=None,
)
@click.option(
    "--batch",
    "max_batch_size",
    help="Maximum batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
)
@click.option(
    "--steps",
    "num_steps",
    help="Number of sampling steps",
    metavar="INT",
    type=click.IntRange(min=1),
    default=18,
    show_default=True,
)
@click.option(
    "--sigma_min",
    help="Lowest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--sigma_max",
    help="Highest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--rho",
    help="Time step exponent",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=7,
    show_default=True,
)
@click.option(
    "--S_churn",
    "S_churn",
    help="Stochasticity strength",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--S_min",
    "S_min",
    help="Stoch. min noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--S_max",
    "S_max",
    help="Stoch. max noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default="inf",
    show_default=True,
)
@click.option(
    "--S_noise",
    "S_noise",
    help="Stoch. noise inflation",
    metavar="FLOAT",
    type=float,
    default=1,
    show_default=True,
)
@click.option(
    "--solver",
    help="Ablate ODE solver",
    metavar="euler|heun",
    type=click.Choice(["euler", "heun"]),
)
@click.option(
    "--disc",
    "discretization",
    help="Ablate time step discretization {t_i}",
    metavar="ddn|vp|ve|iddpm|edm",
    type=click.Choice(["ddn", "vp", "ve", "iddpm", "edm"]),
)
@click.option(
    "--schedule",
    help="Ablate noise schedule sigma(t)",
    metavar="vp|ve|linear",
    type=click.Choice(["vp", "ve", "linear"]),
)
@click.option(
    "--scaling",
    help="Ablate signal scaling s(t)",
    metavar="vp|none",
    type=click.Choice(["vp", "none"]),
)
@click.option(
    "--learn-res",
    help="learn_residual in SDDNOutput",
    metavar="BOOL",
    type=bool,
    default=None,
    show_default=True,
)
@click.option(
    "--skip-exist",
    help="skip-exist",
    metavar="BOOL",
    type=bool,
    default=None,
    show_default=True,
)
@click.option(
    "--sampler",
    help="Guided sampler",
    default=None,
    type=click.Choice(["none", "train", "test", "class", "xflip", "entropy"]),
)
@click.option(
    "--markov",
    help="Markov Sampling [pt_path, 1,0]",
    metavar="PATH|INT",
    type=str,
    default=None,
)
@click.option(
    "--debug",
    help="debug mode",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--fid_ref",
    help="fid ref path e.g. fid-refs/cifar10-32x32.npz",
    metavar="PATH",
    type=str,
    default=None,
    show_default=True,
)
def main(
    network_pkl,
    outdir,
    subdirs,
    seeds,
    class_idx,
    max_batch_size,
    device=torch.device("cuda"),
    learn_res=None,
    skip_exist=True,
    sampler=None,
    markov=None,  # Abandoned, sampling through a priori
    **sampler_kwargs,
):
    (
        """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
        + """
    # Doc string for DDN:
    If len(seeds) > 50000, will eval FID.
    """
    )
    # network_pkl = network_pkl.replace("https://oss.iap.hh-d.brain" + "pp.cn/", "s3://")
    is_s3 = network_pkl.startswith("s3://")
    if outdir is None:
        if is_s3:
            outdir = "/run/generate"
        else:
            outdir = os.path.abspath(os.path.join(network_pkl, "..", "generate"))
        os.makedirs(outdir, exist_ok=True)
    # Visualization Image Path
    visp = (
        (network_pkl + "$").replace(".pkl$", "-vis.png").replace(".pt$", "-vis.png")
        if outdir.endswith("/generate")
        else os.path.abspath(outdir) + "-vis.png"
    )
    eval_dir = visp.replace("-vis.png", "")

    sampler_cmd = "" if (sampler is None or sampler == "none") else sampler
    sampler_prefix = sampler_cmd and (f"sampler.{sampler_cmd}-")
    fid_path = os.path.join(eval_dir, sampler_prefix + "fid.json")
    if skip_exist is None:
        skip_exist = len(seeds) in [100, 50000]

    if markov:
        markov = int(markov) if len(markov) == 1 else markov
        if markov:
            assert markov != 1, "NotImplement!"
            from zero_condition.markov_sampler import MarkovSampler

            markov_sampler = MarkovSampler(markov)

    dist.init()
    if (
        skip_exist
        and (os.path.exists(visp) and len(seeds) == 100)
        or (os.path.exists(fid_path) and len(seeds) == 50000)
    ):
        if torch.distributed.get_rank() == 0:
            print("Vis exists:", visp)
        return

    dirr = os.path.dirname(network_pkl)
    training_options_json = os.path.join(dirr, "training_options.json")
    if os.path.exists(training_options_json):
        train_kwargs = boxx.loadjson(training_options_json)
        boxx.cf.kwargs = train_kwargs.get("kwargs", {})
        if learn_res is None:
            learn_res = train_kwargs.get("kwargs", {}).get(
                "learn_res", "learn.res" in network_pkl
            )
        sddn.DiscreteDistributionOutput.learn_residual = learn_res

    num_batches = (
        (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        if network_pkl.endswith(".pkl"):
            net = pickle.load(f)["ema"].to(device)
        elif network_pkl.endswith(".pt"):
            net = torch.load(f)["net"].to(device)  # 会保存模型代码吗?
            net = net.eval()

    boxx.mg()
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    idx_ks_list = []
    if sampler_cmd:
        from zero_condition.main import (
            ReconstructionDatasetSampler,
            CifarSampler,
            BatchedGuidedSampler,
        )

        if sampler_cmd == "class":
            sampler = CifarSampler(None)
            print("CifarSampler!!!")
        if sampler_cmd == "entropy":
            sampler = CifarSampler(entropy=True)
            print("CifarSampler(entropy=True)!!!")
        if sampler_cmd in ["train", "test"]:
            assert os.path.exists(training_options_json), training_options_json
            # or boxx.cf.debug
            # if boxx.cf.debug:
            #
            # elif
            data_kwargs = train_kwargs["dataset_kwargs"]
            if "cifar" in data_kwargs["path"] and sampler_cmd == "test":
                dataset_guided = None
            else:
                if sampler_cmd == "test":
                    if "ffhq" in data_kwargs["path"]:
                        data_kwargs["path"] = data_kwargs["path"].replace(
                            "ffhq", "celebahq"
                        )
                        data_kwargs["max_size"] = 30000
                    elif "celebahq" in data_kwargs["path"]:
                        data_kwargs["path"] = data_kwargs["path"].replace(
                            "celebahq", "ffhq"
                        )
                        data_kwargs["max_size"] = 70000
                    boxx.cf.kwargs["data"] = data_kwargs["path"]
                # from training import training_loop
                dist.print0(f"dataset_guided-{sampler_cmd}:", data_kwargs)
                dataset_guided = dnnlib.util.construct_class_by_name(**data_kwargs)
            sampler = ReconstructionDatasetSampler(dataset_guided)

        batch_sampler = BatchedGuidedSampler(sampler)
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(
        rank_batches, unit="batch", disable=(dist.get_rank() != 0)
    ):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)
            ]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {
            key: value for key, value in sampler_kwargs.items() if value is not None
        }
        have_ablation_kwargs = any(
            x in sampler_kwargs
            for x in ["solver", "discretization", "schedule", "scaling"]
        )
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        if sampler_kwargs.get("discretization", "ddn") == "ddn":
            sampler_fn = ddn_sampler
            sampler_kwargs["batch_seeds"] = batch_seeds
            if sampler_cmd:
                sampler_kwargs["sampler"] = batch_sampler

            if markov:
                sampler_kwargs["markov_sampler"] = markov_sampler

        images = sampler_fn(
            net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs
        )
        if isinstance(images, dict):
            d, images = images, images["predict"]
            idx_ks = npa(d["idx_ks"]).T  # b, l of np
            idx_ks_list.append(idx_ks)

        # Save images.
        images_np = (
            (images * 127.5 + 128)
            .clip(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = (
                os.path.join(outdir, f"{seed-seed%1000:06d}") if subdirs else outdir
            )
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{seed:06d}.png")
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
            else:
                PIL.Image.fromarray(image_np, "RGB").save(image_path)

    # Done.

    torch.distributed.barrier()
    if dist.get_rank() == 0 and len(seeds) >= 9 and not sampler_cmd:
        # mxs "arrs=npa([imread(pa) for pa in glob('*/*.??g')[:100]]);arrs=arrs.reshape(10,10,*arrs[0].shape);imsave(abspath('.')+'.png', np.concatenate(np.concatenate(arrs,2), 0))"
        example_paths = sorted(glob(outdir + "/**/*.??g", recursive=True))[:100]
        make_vis_img(example_paths, visp)

    if len(seeds) >= 50000:  # or boxx.cf.debug:
        # sync and save latent
        ws = dist.get_world_size()
        ts = {
            rank: -torch.ones(
                ((len(seeds)) // ws + 1, idx_ks_list[0].shape[-1]), dtype=torch.int32
            ).cuda()
            for rank in range(ws)
        }
        for rank, tensor in ts.items():
            if dist.get_rank() == rank:
                idx_ks_ = torch.from_numpy(np.concatenate(idx_ks_list))
                tensor[: len(idx_ks_)] = idx_ks_.to(torch.int32)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.broadcast(tensor, src=rank)
        if dist.get_rank() == 0:
            ddn_latents = torch.cat(tuple(ts.values()))
            ddn_latents = ddn_latents[ddn_latents[:, 0] >= 0].short().cpu()
            os.makedirs(eval_dir, exist_ok=True)
            ddn_latents_path = os.path.join(
                eval_dir,
                sampler_prefix
                + f"ddn_latents_l{len(ddn_latents[0])}_n{len(ddn_latents)}.pt",
            )
            print("Save DDN latents to:", ddn_latents_path)
            torch.save(ddn_latents, ddn_latents_path)
            if sampler_cmd == "train":
                torch.save(
                    ddn_latents, os.path.join(eval_dir, "train_seqs_for_markov.pt")
                )

            # 1/0

        import fid

        if sampler_kwargs.get("fid_ref", None):
            fid_argkws = dict(
                ref_path=sampler_kwargs["fid_ref"],
                image_path=outdir,
                num_expected=min(len(seeds), 50000),
                seed=0,
                batch=max_batch_size,
            )
            dist.print0("fid_argkws:", fid_argkws)
            fid = fid.calc_fid(**fid_argkws)
        else:
            dist.print0("fid_argkws:", fid_argkws)
            fid = fid_argkws = {
                "fid": -1,
                "info": "no fid ref",
            }
        if dist.get_rank() == 0:
            kimg = ([-1] + boxx.findints(os.path.basename(network_pkl)))[-1]
            os.makedirs(eval_dir, exist_ok=True)
            tmp_tar = "/run/ddn.tar"
            tar_path = os.path.join(eval_dir, sampler_prefix + "sample-example.tar")
            print("Saving example images tar to:", tar_path)
            example_paths = sorted(glob(outdir + "/**/*.??g", recursive=True))[:100]
            boxx.zipTar(example_paths, tmp_tar)
            copy_file = lambda src, dst: open(dst, "wb").write(open(src, "rb").read())
            copy_file(tmp_tar, tar_path)
            make_vis_img(
                example_paths, os.path.join(eval_dir, sampler_prefix + "vis.png")
            )

            boxx.savejson(
                dict(
                    fid=fid["fid"],
                    path=network_pkl,
                    kimg=kimg,
                    kwargs=boxx.cf.kwargs,
                    fid_argkws=fid_argkws,
                ),
                fid_path,
            )
            boxx.savejson(
                dict(fid=fid["fid"], path=network_pkl, kimg=kimg),
                os.path.join(eval_dir, sampler_prefix + "fid-%.3f" % fid["fid"]),
            )
    dist.print0("Done.")
    if boxx.cf.debug:
        sdd = net.model.block_32x32_1.ddo.sdd
        sdd.plot_dist()
    boxx.mg()


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath("."))
    torch.distributed.GroupMember.WORLD = None
    import boxx
    from boxx.ylth import *
    from ddn_utils import debug, argkv

    print("debug", debug)
    if not debug:
        main()
    else:
        boxx.cf.debug = True
        main(
            [
                "--seeds=0-15",
                # "--network=../asset/v12_augment0-00000-ffhq-64x64-outputk8_learn.res-007526.pkl",
                # "--network=../asset/v13_new.setting-00000-ffhq64-fp16-dropout0-200000.pkl",
                # "--network=cifar10-ddn.pkl",
                "--network=../asset/v15-00023-ffhq-64x64-blockn64_outputk64_chain.dropout0.05-shot-117913.pkl",
                # "--network=../asset/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl",
                # "--network=exps/cifar10-ddn.pkl",
                "--outdir=/tmp/gen_ddn",
                "--batch=3",
                # "--sampler=test",
                # "--markov=../asset/sampler.train-ddn_latents_l63_n50000.pt",
            ]
        )

        # %%
        dataset_obj = dnnlib.util.construct_class_by_name(
            **{
                "class_name": "training.dataset.ImageFolderDataset",
                "path": "datasets/cifar10-32x32.zip",
                "use_labels": False,
                "xflip": False,
                "cache": True,
                "resolution": 32,
                "max_size": 50000,
            }
        )
        d = {
            "target": tht(
                np.concatenate([dataset_obj[i + 15][0][None] for i in range(3)])
            )
            .cuda()
            .to(torch.float32)
            / 127.5
            - 1
        }
        d = net(d)
        showd = lambda d_=d: show(
            d_["predicts"][3:],
            d_.get("target"),
            d_["predict"],
            lambda x: (x * 127.5 + 128).clip(0, 255).astype(np.uint8),
            tprgb,
        )
        showd(d)


# ----------------------------------------------------------------------------
