#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 12:56:20 2025

@author: DIYer22
"""

import os
import boxx
import sys
import PIL.Image
import numpy as np
import torch
import threading


from boxx import *

sys.path.append("..")
with inpkg():
    from .main import *
    from ..ddn_utils import *
    from ..training.dataset import ImageFolderDataset
    from .main import topk_sample, BatchedGuidedSampler, MultiGuidedSampler


class DistanceSamplerWithAlphaChannelTopk:
    def __init__(self, target, topk=2, distance_type="l1"):
        # TODO try using weighted HSV distance
        self.raw = target  # guided raw format, here is (4, h, w) of torch.cuda.FloatTensor [-1, 1]
        if target.shape[0] == 3:  # process target in __init__
            target = torch.cat([target, target[:1] * 0 + 1], dim=0)
        self.target = target[None]
        self.hw = target.shape[-2:]
        self.topk = topk
        self.distance_type = distance_type

    def __call__(self, dic):
        rgbs = dic["rgbs"]  # K RGB outputs for 1 data, not whole batch
        k, c, h, w = rgbs.shape
        if h != self.hw[0] or w != self.hw[1]:
            # resize rgbs to target size, not inverse, to avoid color in transparent area influence the sampling
            # alternative: resize target to rgbs size may faster, but may need Median Downsample Pooling
            rgbs = nn.functional.interpolate(rgbs, self.hw, mode="bilinear")
        alpha = self.target[:, 3:].clip(-1, 1) / 2 + 0.5
        target = self.target[:, :3]
        if self.distance_type == "l1":
            loss_map = torch.abs(rgbs - target)
        elif self.distance_type == "l2":
            loss_map = (rgbs - target) ** 2
        else:
            raise ValueError(f"Invalid distance type: {self.distance_type}")
        probs = nn.functional.softmax(
            -(loss_map * alpha).sum([-1, -2]).mean(-1) / (alpha.sum() + 1e-6),
            0,
        )
        # probs = nn.functional.softmax(-((rgbs - self.target) ** 2).mean([-1, -2, -3]), 0)
        return dict(
            probs=probs,  # K probabilities for sampling
            idx_k=topk_sample(probs, self.topk),  # sampled index
            condition0=self.target,
            condition_source0=self.raw,
        )


def crop_and_resize(img: np.ndarray, out_hw=(256, 256)) -> np.ndarray:
    """
    Center-crop an image to match the aspect ratio of `out_hw`
    and then resize it. Works for any rectangle, not just squares.
    """
    target_h, target_w = out_hw
    target_ratio = target_w / target_h

    h, w = img.shape[:2]
    in_ratio = w / h
    if (h, w) == out_hw:
        return img
    # Decide which dimension to crop
    if np.isclose(in_ratio, target_ratio, rtol=0, atol=1e-6):
        crop = img  # already the right ratio
    elif in_ratio > target_ratio:
        # Too wide → crop width
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        crop = img[:, x0 : x0 + new_w]
    else:
        # Too tall → crop height
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        crop = img[y0 : y0 + new_h, :]

    # Resize with high-quality Lanczos
    pil_img = PIL.Image.fromarray(crop)
    resized = pil_img.resize(out_hw, PIL.Image.LANCZOS)

    return np.asarray(resized, dtype=img.dtype)


class DDNInference:
    def __init__(self, weight_path, hw=(256, 256)):
        self.net = load_net(weight_path)
        self.hw = hw

    def inference(self, inf_arg):
        # d_init = dict(condition_source=condition_source)
        # d = self.net(d_init)
        pass

    def process_np_img(self, img):
        # resize with PIL
        if img.shape[0] != self.hw[0] or img.shape[1] != self.hw[1]:
            img = crop_and_resize(img, self.hw)
        if img.ndim == 2:
            img = np.concatenate([img[..., None]] * 3, -1)
        return uint8_to_tensor(img)

    def coloring_demo_inference(
        self, condition_rgb, n_samples=1, guided_rgba=None, clip_prompt=None
    ):  # all Numpy
        condition_source = self.process_np_img(condition_rgb)
        samplers = []
        if guided_rgba is not None and guided_rgba[..., -1].sum() > 1:
            guided_rgba = self.process_np_img(guided_rgba)
            if "align color brightness" and 0:
                print("align color brightness")
                guided_normalized_tensor = (guided_rgba[:3] + 1) / 2
                brightness_guided = (guided_normalized_tensor).mean(0, keepdim=True)
                brightness_condition = ((condition_source + 1) / 2).mean(
                    0, keepdim=True
                )
                new_guided_normalized_tensor = (
                    guided_normalized_tensor
                    * brightness_condition
                    / (brightness_guided + 1e-5)
                )
                guided_rgba[:3] = new_guided_normalized_tensor.clip(0, 1) * 2 - 1
            rgba_sampler = DistanceSamplerWithAlphaChannelTopk(guided_rgba)
            samplers.append(rgba_sampler)
        if clip_prompt is not None and clip_prompt not in ["", "null"]:
            clip_sampler = CLIPSampler(clip_prompt, keep_model_in_memory=True)
            samplers.append(clip_sampler)
        d_init = dict(condition_source=torch.cat([condition_source[None]] * n_samples))
        if len(samplers) == 1:
            batch_sampler = BatchedGuidedSampler(samplers[0])
            d_init["sampler"] = batch_sampler
        elif len(samplers) > 1:
            batch_sampler = MultiGuidedSampler({s: 1 / len(samplers) for s in samplers})
            d_init["sampler"] = batch_sampler
        # tree(["d_init", d_init, guided_rgba])
        d = self.net(d_init)
        stage_last_predicts = {
            "%sx%s" % pred.shape[-2:]: pred for pred in d["predicts"]
        }
        stage_last_predicts_np = {
            k: list(t2rgb(v)) for k, v in stage_last_predicts.items()
        }
        d["stage_last_predicts_np"] = stage_last_predicts_np
        boxx.mg()
        # shows(t2rgb(d["predict"]), png=True)
        return d


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    weight_path = "../../ddn_asset/v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-200000.pkl"
    test_img_path = "../../ddn_asset/ffhq_example/FFHQ-test4.png"

    if not os.path.exists(weight_path) or not os.path.exists(test_img_path):
        weight_path = hf_hub_download(
            repo_id="diyer22/ddn_asset",
            filename="v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-200000.pkl",
        )
        test_img_path = hf_hub_download(
            repo_id="diyer22/ddn_asset", filename="ffhq_example/FFHQ-test4.png"
        )
    # ddn = sys._getframe(6).f_globals.get("ddn")
    ddn = None
    if ddn is None:
        print("loading weight")
        ddn = DDNInference(weight_path)
        print(ddn.net.model.table())
    condition_rgb = imread(test_img_path)
    guided_rgba = np.uint8(condition_rgb[:] * 0 + [[(255, 0, 0)]])
    mask = np.ones_like(guided_rgba)[..., :1] * 0
    mask[: len(mask) // 10 :, : len(mask) // 10] = 255
    guided_rgba_np = np.concatenate([guided_rgba // 2, mask], axis=-1)
    # guided_rgba = None
    clip_prompt = "Portrait with light blue background"
    clip_prompt = "Portrait of retro tones, warm"
    clip_prompt = "Blue eyes, red skin, pink background"
    clip_prompt = "Wasteland style portrait"
    # clip_prompt = "Portrait, healthy skin tone, dark brown hair"
    clip_prompt = ""
    d = ddn.coloring_demo_inference(
        condition_rgb, n_samples=6, guided_rgba=guided_rgba_np, clip_prompt=clip_prompt
    )
    shows(
        condition_rgb,
        guided_rgba_np,
        t2rgb(d["predict"]),
        png=True,
    )
    # 256png 104.8KB
    # 256jpg 10.7KB
