# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
import boxx
import numpy as np
import torch
import random
import math

if __name__ == "__main__":
    from boxx.ylth import *
    import sys

    sys.path.append("..")
# with boxx.impt(".."):
import torch_utils
from torch_utils import persistence, misc
from torch.nn.functional import silu

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        w_pad = int(w_pad)
        f_pad = int(f_pad)

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch


@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[
            1,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = (
            Linear(in_features=label_dim, out_features=noise_channels, **init)
            if label_dim
            else None
        )
        self.map_augment = (
            Linear(
                in_features=augment_dim, out_features=noise_channels, bias=False, **init
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=noise_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]
        # boxx.g()/0
        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )
        boxx.mg()
        boxx.cf.debug and boxx.g()

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels  # batch, class of one hot (3, 10) of torch.cuda.FloatTensor @ cuda:0
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            # boxx.increase("lable debug") and boxx.g()/0
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


import sddn
from sddn import DiscreteDistributionOutput


@persistence.persistent_class
class UNetBlockWoEmb(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        # self.affine = Linear(
        #     in_features=emb_channels,
        #     out_features=out_channels * (2 if adaptive_scale else 1),
        #     **init,
        # )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb=None):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        # params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        # if self.adaptive_scale:
        #     scale, shift = params.chunk(chunks=2, dim=1)
        #     x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        # else:
        #     x = silu(self.norm1(x.add_(params)))
        x = silu(self.norm1(x))
        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


@persistence.persistent_class
class WarpDictIO(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.warp_dict = block

    def forward(self, d):
        d["feat_last"] = self.warp_dict.forward(d["feat_last"])
        return d


@persistence.persistent_class
class SongUNetInputDict(SongUNet):
    def forward(self, d):
        d["feat_last"] = super().forward(
            d["feat_last"],
            d.get("noise_labels"),
            d.get("class_labels"),
            d.get("augment_labels"),
        )
        # def forward(self, x, noise_labels, class_labels, augment_labels=None):
        return d


@persistence.persistent_class
class UpBlock(UNetBlockWoEmb):
    def forward(self, d):
        d["feat_last"] = super().forward(d["feat_last"])
        b, c, h, w = d["feat_last"].shape
        if "predict" in d:
            d["predict"] = torch.nn.functional.interpolate(
                d["predict"], (h, w), mode="bilinear"
            )
        if "feat_leak" in d:
            d["feat_leak"] = torch.nn.functional.interpolate(
                d["feat_leak"], (h, w), mode="bilinear"
            )

        return d


@persistence.persistent_class
class DiscreteDistributionBlock(torch.nn.Module):
    short_plus = True
    # short_plus = False

    def __init__(
        self,
        block,
        k=64,
        output_size=None,
        in_c=None,
        out_c=None,
        predict_c=3,
        loss_func=None,
        distance_func=None,
        leak_choice=True,
        input_dict=False,
    ):
        super().__init__()
        self.block = block
        block_first, block_last = (
            (block[0], block[-1])
            if isinstance(block, torch.nn.Sequential)
            else (block, block)
        )
        self.in_c = (
            in_c
            or getattr(block_first, "in_c", None)
            or getattr(block_first, "in_channels", None)
        )
        self.out_c = (
            out_c
            or getattr(block_last, "out_c", None)
            or getattr(block_last, "out_channels", self.in_c)
        )
        self.predict_c = predict_c
        self.leak_choice = leak_choice
        self.input_dict = input_dict

        # TODO replace choice and leak conv to short plus
        if not self.short_plus:
            self.choice_conv1x1 = sddn.Conv2dMixedPrecision(
                predict_c, self.in_c, (1, 1), bias=False
            )
            if leak_choice:
                self.leak_conv1x1 = sddn.Conv2dMixedPrecision(
                    predict_c, self.in_c, (1, 1), bias=False
                )
        self.ddo = DiscreteDistributionOutput(
            k,
            last_c=self.out_c,
            predict_c=predict_c,
            size=output_size,
            loss_func=loss_func,
            distance_func=distance_func,
            leak_choice=leak_choice,
        )
        self.output_size = output_size

    def forward(self, d=None, condition_process=None):
        d = d if isinstance(d, dict) else {"batch_size": 1 if d is None else len(d)}
        if "target" in d:
            batch_size = len(d["target"])
        else:
            batch_size = d.get("batch_size", 1)
        inp = d.get("feat_last")
        predict = d.get("predict")
        feat_leak = d.get("feat_leak")
        if inp is None:  # init d
            # inp = torch.cat(
            #     [torch.linspace(-1, 1, self.in_c).reshape(1, self.in_c, 1, 1)]
            #     * batch_size
            # ).cuda()
            # predict = torch.cat(
            #     [torch.linspace(-1, 1, self.predict_c).reshape(1, self.predict_c, 1, 1)]
            #     * batch_size
            # ).cuda()

            # Destructive changes to transfer learning
            inp = sddn.build_init_feature(
                (batch_size, self.in_c, self.output_size, self.output_size)
            ).cuda()
            predict = torch.zeros(
                (batch_size, self.predict_c, self.output_size, self.output_size)
            ).cuda()

            if boxx.cf.get("kwargs", {}).get(
                "fp16",
            ):
                inp, predict = inp.half(), predict.half()
            feat_leak = predict
        d["feat_last"] = inp
        b, c, h, w = inp.shape
        if not hasattr(self, "choice_conv1x1"):
            # boxx.g()/0
            # inp[:, :self.predict_c].add_(predict)
            inp = inp + torch.nn.functional.pad(
                predict, (0, 0, 0, 0, 0, c - self.predict_c)
            )
            if self.leak_choice:
                # inp[:, -self.predict_c:].add_(feat_leak)
                inp = inp + torch.nn.functional.pad(
                    feat_leak, (0, 0, 0, 0, c - self.predict_c, 0)
                )
            if condition_process:
                stage_condition = condition_process(d)
                if stage_condition is not None:
                    condc = stage_condition.shape[1]
                    cond_start = c // 2 - condc // 2
                    inp = inp + torch.nn.functional.pad(
                        stage_condition,
                        (0, 0, 0, 0, cond_start, c - cond_start - condc),
                    )

        else:
            inp = inp + self.choice_conv1x1(predict)
            if self.leak_choice:
                inp = inp + self.leak_conv1x1(feat_leak)
        if self.input_dict:
            d["feat_last"] = inp
            d = self.block(d)
        else:
            d["feat_last"] = self.block(inp)
        d = self.ddo(d)
        # g()/0
        return d


def get_channeln(scalei):
    channeln = 2 ** (13 - scalei)
    return min(max(4, channeln), 256)


def get_outputk(scalei, predict_c=3):
    # if predict_c == 1:
    k = 4 * get_channeln(scalei)
    k = min(max(16, k), 1024)
    if predict_c == 3:
        k = k // 2
    # return 3
    k = min(boxx.cf.get("kwargs", {}).get("max_outputk", k), k)
    return k


def get_blockn(scalei):
    scalei_to_blockn = {
        0: 1,
        1: 2,
        2: 4,
        3: 8,
        4: 16,
        5: 32,
        6: 64,
    }
    if scalei not in scalei_to_blockn:
        scalei_to_blockn[scalei] = max(scalei_to_blockn.values())
    blockn = scalei_to_blockn[scalei]
    blockn = min(boxx.cf.get("kwargs", {}).get("max_blockn", blockn), blockn)
    return blockn


class ClassEmbeding(torch.nn.Module):
    def __init__(self, classn=10, bit_each_channel=2, class0_is_zeros=True):
        super().__init__()
        self.emb = (
            self.get_class_embeding(
                classn=classn,
                bit_each_channel=bit_each_channel,
                class0_is_zeros=class0_is_zeros,
            )
            .requires_grad_(False)
            .cuda()
        )
        self.classn = classn
        self.emb_length = self.emb.shape[-1]

    def __call__(self, label):
        if label.dtype.is_floating_point:
            label = label.argmax(-1)  # (b, n) one hot => (b,) long
        label_embs = self.emb.to(label.device)[label]  # (b, emb_length)
        return label_embs
        # label_embs_4dim = label_embs[...,None,None]  # (b, emb_length, 1, 1)

    @staticmethod
    def get_class_embeding(classn=10, bit_each_channel=2, class0_is_zeros=True):
        import math

        num_base = 1 << bit_each_channel
        emb_length = int(math.log2(classn - 1) / bit_each_channel) + 1
        emb = torch.zeros([classn, emb_length])
        bits_to_v = dict(
            zip(
                [
                    ("0" * bit_each_channel + bin(i)[2:])[-bit_each_channel:]
                    for i in range(num_base)
                ],
                torch.linspace(-1, 1, num_base),
            )
        )
        for classi in range(classn):
            if class0_is_zeros and classi == 0:  # skip class0, class0 as zereos
                continue
            bin_str = bin(classi)[2:]
            bits = (emb_length * bit_each_channel - len(bin_str)) * "0" + bin_str
            emb[classi] = torch.as_tensor(
                [
                    bits_to_v[
                        bits[
                            i * bit_each_channel : i * bit_each_channel
                            + bit_each_channel
                        ]
                    ]
                    for i in range(emb_length)
                ]
            )
        return emb  # (classn, emb_length)


class ConditionProcess(torch.nn.Module):
    def __init__(self, condition_type=None, input_at="stage_begin"):
        assert input_at in ["stage_begin", "every_level"], input_at
        super().__init__()
        self.input_at = input_at
        self.condition_type = condition_type
        if condition_type == "edge":
            self.sobel_kernel_horizontal = (
                torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[
                    None, None
                ]
                .requires_grad_(False)
                .cuda()
            )
            self.sobel_kernel_vertical = (
                torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])[
                    None, None
                ]
                .requires_grad_(False)
                .cuda()
            )
        if condition_type.startswith("class"):
            self.classn = int(condition_type[len("class") :])
            self.class_emb = ClassEmbeding(self.classn)

    def init_dict(self, d):
        ct = self.condition_type
        if not ct or "condition" in d:
            return d
        dtype = d["feat_last"].dtype
        with torch.no_grad():
            condition_source = d.get("condition_source", d.get("target"))
            # assert condition_source is not None or ct=="class", f"Condition type is {ct}, please provide condition_source"
            d["condition"] = []
            if ct.startswith("resize"):
                self.resized_size = int(ct[len("resize") :])
                h, w = condition_source.shape[-2:]
                resized = torch.nn.functional.interpolate(
                    condition_source,
                    (self.resized_size, self.resized_size),
                    mode="area",
                )
                resize_back = torch.nn.functional.interpolate(
                    resized, (h, w), mode="bilinear"
                )
                d["condition"].append(resize_back)
                d["condition_resized"] = resized
            if ct == "color":  # TODO do random per batch rather than per iter
                # d["condition"].append(condition_source.mean(-3, keepdim=True))
                if self.training:
                    # Generate random weights for RGB channels that sum to 1
                    weights = torch.rand(3, device=condition_source.device)
                    weights = weights / weights.sum()  # normalize weights to sum to 1

                    # Reshape weights for broadcasting: [3] -> [1, 3, 1, 1]
                    weights = weights.reshape(1, 3, 1, 1)

                    # Apply weighted average
                    weighted_grayscale = (condition_source * weights).sum(
                        dim=-3, keepdim=True
                    )
                    d["condition"].append(weighted_grayscale)
                else:
                    # During evaluation, use equal weights (mean)
                    d["condition"].append(condition_source.mean(-3, keepdim=True))
            if ct == "edge":
                grey = condition_source.mean(-3, keepdim=True)
                output_horizontal = torch.nn.functional.conv2d(
                    grey, self.sobel_kernel_horizontal
                )
                output_vertical = torch.nn.functional.conv2d(
                    grey, self.sobel_kernel_vertical
                )

                magnitude = torch.sqrt(output_horizontal**2 + output_vertical**2)
                thre = 0.75
                if self.training:
                    thre += random.random() / 2 - 0.25
                edge = (magnitude > thre).to(dtype) * 2 - 1
                # shows-[img,edges, cv2.Canny(img, 100, 200), edges>1.5]
                d["condition"].append(edge)
            if ct.startswith("class"):
                # When to append conditions is better. Note class_emb could be empty when inference, so should add to last of conditions
                if d.get("class_labels") is None:
                    d["class_labels"] = torch.zeros(
                        (d["feat_last"].shape[0], self.classn)
                    ).cuda()
                    d["class_labels"][:, 0] = 1
                class_emb_4dim = self.class_emb(d["class_labels"])[
                    ..., None, None
                ]  # (b, emb_length, 1, 1)
                d["condition"].append(class_emb_4dim)
            d["condition"] = torch.cat([c for c in d["condition"]], 1).to(dtype)
        return d

    def forward(self, d):
        if self.input_at == "stage_begin":
            if "last_level_size" not in d:
                d = self.init_dict(d)
            size = d["feat_last"].shape[-1]
            if size != d.get("last_level_size"):
                d["last_level_size"] = size
                return torch.nn.functional.interpolate(
                    d["condition"], (size, size), mode="area"
                )
        else:
            raise NotImplementedError()


@persistence.persistent_class
class PHDDNHandsDense(
    torch.nn.Module
):  # PyramidHierarchicalDiscreteDistributionNetwork
    def __init__(
        self,
        img_resolution=32,  # Image resolution at input/output.
        in_channels=3,  # Number of color channels at input.
        out_channels=3,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[
            1,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        # noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )
        self.label_dim = label_dim

        condition_type = boxx.cf.get("kwargs", {}).get("condition")
        if condition_type:
            if condition_type == "class":
                assert condition_type and label_dim, (condition_type, label_dim)
                condition_type += str(label_dim)
            self.condition_process = ConditionProcess(condition_type)
        self.condition_type = condition_type

        self.scalen = int(np.log2(img_resolution))

        self.module_names = []
        self.scale_to_module_names = {}  # dict for if scale is float, like 1.5
        self.scale_to_repeatn = {}

        # DiscreteDistributionOutput.learn_residual = False
        # DiscreteDistributionBlock.short_plus = False

        if "hands design":
            """
            手工设计网络, 原则:
                - 小 scale:
                    - 背景: 小 scale 即低频信息, 低频信息一定是可以无损压缩的!,  所以需要大算力和表征空间的限制来让网络压缩低频信息. 但表征空间需要大于该尺度的实际信息量!
                    - 减少 k, 增大 repeat 和 blockn * channeln, 因为 k 要多复用, 避免学不会和过拟合, 低维能有效分岔, 需要更多算力
                - 大 scale
                    - 高频信息难以无损压缩, 但可以通过更多的表示空间/采样 + 更多的采样来逼近高频信号, 以减缓平均模糊现象
                    - 巨大的 k, 不要算力. 考更多 k 带来空间和随机性, 符合高频信号的随机性, 和低算力需求特性
            """
            # scale_to_channeln = [256, 256, 256, 256, 256, 256, 128]
            # scale_to_blockn = [1, 2, 4, 8, 8, 8, 8]
            # scale_to_repeatn = [1] * 6
            # scale_to_outputk = [512, 512, 512, 512, 512, 512, 256]

            # 1, 2, 4, 8, 16, 32, 64
            scale_to_channeln = [256, 256, 256, 256, 128, 64, 32]
            scale_to_blockn = [1, 8, 16, 16, 8, 4, 3]
            scale_to_repeatn = [3, 10, 10, 10, 10, 5, 2]
            scale_to_outputk = [64, 16, 16, 16, 64, 512, 512]
            # if boxx.cf.debug:
            #     scale_to_channeln = [4, 8] * 7
            # get_channeln = lambda scalei: scale_to_channeln[scalei]
            # get_blockn = lambda scalei: scale_to_blockn[scalei]
            # get_outputk = lambda scalei: scale_to_outputk[scalei]
            # get_repeatn = lambda scalei: scale_to_repeatn[scalei]
            # self.scale_to_repeatn = dict(enumerate(scale_to_repeatn))

        def set_block(name, block):
            self.module_names.append(name)
            setattr(self, name, block)
            self.scale_to_module_names[scalei] = self.scale_to_module_names.get(
                scalei, []
            ) + [name]
            return block

        start_size = boxx.cf.get("kwargs", {}).get("start_size", 1)
        blockn_times = boxx.cf.get("kwargs", {}).get("blockn_times", 1)
        self.scalis = range(int(math.log2(start_size)), self.scalen + 1)

        last_scalei = self.scalis[0]
        for scalei in self.scalis:
            size = 2**scalei
            channeln = get_channeln(scalei)
            last_channeln = get_channeln(scalei - 1)
            k = get_outputk(scalei)
            if last_scalei != scalei:
                # up block
                block_up = UNetBlockWoEmb(
                    in_channels=last_channeln,
                    out_channels=channeln,
                    up=True,
                    **block_kwargs,
                )
                set_block(
                    f"block_{size}x{size}_0_up",
                    DiscreteDistributionBlock(block_up, k, output_size=size),
                )
            else:  # scale0 only 1 block
                block = UNetBlockWoEmb(channeln, channeln, **block_kwargs)
                set_block(
                    f"block_{size}x{size}_0",
                    DiscreteDistributionBlock(block, k, output_size=size),
                )
                if not scalei:  # 1 blockn for scalei==0(1x1)
                    continue
            cin = channeln
            blockn = int(round(get_blockn(scalei) * blockn_times))
            for block_count in range(1, blockn):
                block = UNetBlockWoEmb(cin, channeln, **block_kwargs)
                set_block(
                    f"block_{size}x{size}_{block_count}",
                    DiscreteDistributionBlock(block, k, output_size=size),
                )
                cin = channeln
        self.refiner_repeatn = (
            3 if boxx.cf.debug else boxx.cf.get("kwargs", {}).get("refinern", 0)
        )
        refiner_outputk = 4  # 由于大体结构已经确定, 希望网络只做 debulr 操作, 所以理论上 outputk 为 1 就可以了
        if self.refiner_repeatn:
            unet = SongUNetInputDict(
                img_resolution=img_resolution,
                in_channels=channeln,  # input is channeln.
                out_channels=channeln,  # output is channeln.
                label_dim=label_dim,
                augment_dim=augment_dim,
                model_channels=model_channels,
                channel_mult=channel_mult,
                channel_mult_emb=channel_mult_emb,
                num_blocks=num_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                label_dropout=label_dropout,
                embedding_type=embedding_type,
                channel_mult_noise=channel_mult_noise,
                encoder_type=encoder_type,
                decoder_type=decoder_type,
                resample_filter=resample_filter,
            )
            self.refiner = DiscreteDistributionBlock(
                unet,
                refiner_outputk,
                output_size=img_resolution,
                in_c=channeln,
                out_c=channeln,
                predict_c=out_channels,
                input_dict=True,
            )

    def forward(self, d=None, _sigma=None, labels=None):
        # labels's shape (batch, class) of one hot (3, 10) of torch.cuda.FloatTensor @ cuda:0
        # class0 = zeros, if not has lables, then condition is class0

        if isinstance(d, torch.Tensor):
            d = {"target": d}
        elif d is None:
            d = {"batch_size": 1}
        assert isinstance(d, dict), d

        # d = d if isinstance(d, dict) else {"batch_size": 1 if d is None else len(d)}
        if self.label_dim and labels is not None:
            d["class_labels"] = labels
        for scalei in self.scalis:
            for repeati in range(self.scale_to_repeatn.get(scalei, 1)):
                for module_idx, name in enumerate(self.scale_to_module_names[scalei]):
                    # print(name)
                    if module_idx == 0 and repeati != 0:
                        # skip first moule (up sample) when repeat
                        continue
                    module = getattr(self, name)
                    d = module(
                        d, condition_process=getattr(self, "condition_process", None)
                    )
        feat = d["feat_last"]
        batch_size = feat.shape[0]
        for repeati in range(self.refiner_repeatn):
            d["noise_labels"] = torch.Tensor(
                [(repeati / max(self.refiner_repeatn - 1, 1)) * 2 - 1] * batch_size
            ).to(feat)
            # print(d["noise_labels"])
            # print(d.get("augment_labels"))
            # print("repeati", repeati)
            d = self.refiner(
                d, condition_process=getattr(self, "condition_process", None)
            )
        # boxx.tree-d
        # print(repeati)
        return d

    def table(
        self,
    ):
        times = 1
        mds = []
        for name in self.module_names:
            m = getattr(self, name)
            k = m.ddo.k if hasattr(m, "ddo") else 1
            c = (m.in_c, m.out_c) if hasattr(m, "in_c") else (None, None)
            size = m.output_size if hasattr(m, "output_size") else size
            repeat = (
                self.scale_to_repeatn.get(int(np.log2(size)), 1)
                if hasattr(m, "output_size")
                else 1
            )
            times *= k * repeat
            log2 = math.log2(times)
            row = dict(
                name=name,
                size=size,
                c=c,
                k=k,
                repeat=repeat,
                log2=log2,
                log10=math.log10(times),
            )
            mds.append(row)
        return boxx.Markdown(mds)

    def get_sdds(self):
        sdds = []
        for name in self.module_names:
            m = getattr(self, name)
            sdds.append(m.ddo.sdd)
        return sdds


@persistence.persistent_class
class PHDDNHandsSparse(
    PHDDNHandsDense
):  # PyramidHierarchicalDiscreteDistributionNetwork
    def __init__(
        self,
        img_resolution=32,  # Image resolution at input/output.
        in_channels=3,  # Number of color channels at input.
        out_channels=3,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[
            1,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        # DiscreteDistributionOutput.learn_residual = False
        # DiscreteDistributionBlock.short_plus = False

        torch.nn.Module.__init__(self)
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        self.scalen = int(np.log2(img_resolution))

        self.module_names = []
        self.scale_to_module_names = {}  # dict for if scale is float, like 1.5
        self.scale_to_repeatn = {}

        if "hands design":
            """
            手工设计网络, 原则:
                - 小 scale:
                    - 背景: 小 scale 即低频信息, 低频信息一定是可以无损压缩的!,  所以需要大算力和表征空间的限制来让网络压缩低频信息. 但表征空间需要大于该尺度的实际信息量!
                    - 减少 k, 增大 repeat 和 blockn * channeln, 因为 k 要多复用, 避免学不会和过拟合, 低维能有效分岔, 需要更多算力
                - 大 scale
                    - 高频信息难以无损压缩, 但可以通过更多的表示空间/采样 + 更多的采样来逼近高频信号, 以减缓平均模糊现象
                    - 巨大的 k, 不要算力. 考更多 k 带来空间和随机性, 符合高频信号的随机性, 和低算力需求特性
            """
            # scale_to_channeln = [256, 256, 256, 256, 256, 256, 128]
            # scale_to_blockn = [1, 2, 4, 8, 8, 8, 8]
            # scale_to_repeatn = [1] * 6
            # scale_to_outputk = [512, 512, 512, 512, 512, 512, 256]

            # 1, 2, 4, 8, 16, 32, 64
            scale_to_channeln = [256, 256, 256, 256, 128, 64, 32]
            scale_to_blockn = [4, 8, 16, 16, 8, 5, 4]
            scale_to_repeatn = [2, 10, 10, 10, 10, 6, 3]
            scale_to_outputk = [64, 32, 32, 32, 64, 512, 512]
            if boxx.cf.debug:
                scale_to_channeln = [4, 8] * 7
            get_channeln = lambda scalei: scale_to_channeln[scalei]
            get_blockn = lambda scalei: scale_to_blockn[scalei]
            get_outputk = lambda scalei: scale_to_outputk[scalei]
            get_repeatn = lambda scalei: scale_to_repeatn[scalei]
            self.scale_to_repeatn = dict(enumerate(scale_to_repeatn))

        def set_block(name, block):
            self.module_names.append(name)
            setattr(self, name, block)
            self.scale_to_module_names[scalei] = self.scale_to_module_names.get(
                scalei, []
            ) + [name]
            return block

        for scalei in range(self.scalen + 1):
            size = 2**scalei
            channeln = get_channeln(scalei)
            last_channeln = get_channeln(scalei - 1)
            k = get_outputk(scalei)
            if scalei:
                # up block
                block_up = UpBlock(
                    in_channels=last_channeln,
                    out_channels=channeln,
                    up=True,
                    **block_kwargs,
                )
                set_block(
                    f"block_{size}x{size}_up",
                    (block_up),
                )
            else:  # scale0 no up
                pass
            blocks = [
                UNetBlockWoEmb(channeln, channeln, **block_kwargs)
                for block_count in range(0, get_blockn(scalei))
            ]
            blocks = torch.nn.Sequential(*blocks)
            dd_blocks = DiscreteDistributionBlock(blocks, k, output_size=size)
            set_block(
                f"blocks_{size}x{size}",
                dd_blocks,
            )

    def forward(self, d=None, _sigma=None, labels=None):
        for scalei in range(self.scalen + 1):
            for repeati in range(self.scale_to_repeatn.get(scalei, 1)):
                for module_idx, name in enumerate(self.scale_to_module_names[scalei]):
                    if name.endswith("_up") and repeati != 0:
                        # skip first moule (up sample) when repeat
                        continue
                    # print(name)
                    module = getattr(self, name)
                    d = module(d)
        return d


PHDDN = PHDDNHandsDense
# PHDDN = PHDDNHandsSparse
if __name__ == "__main__":
    img_resolution, in_channels, out_channels = 32, 3, 3
    target = torch.zeros((2, 3, 32, 32)).cuda()
    torch.autograd.set_detect_anomaly(True)

    if "SongUNet" and 0:
        net = SongUNet(img_resolution, in_channels, out_channels, num_blocks=2).cuda()
        params = [
            torch.randn(shape).cuda().requires_grad_(True)
            for shape in [(2, 3, 32, 32), (2,), (2, 0), (2, 9)]
        ]
        params[0] = (params[0] * 0).half()
        net = net.train()
        with boxx.timeit("SongUNet"):
            out = misc.print_module_summary(net, params, max_nesting=1)
        1 / 0
    if "DDN":
        img_resolution = 32
        boxx.cf.debug = True
        net = PHDDN(img_resolution, in_channels, out_channels).cuda()
        params = dict(target=target)
        # d = net(params)
        # d = misc.print_module_summary(net.eval(), [], max_nesting=1)
        d = net.train()(params)
        loss = sum(d["losses"])
        # loss = d["losses"][0]
        loss.backward()
        print(net.table())
        tree - d

if 0:
    # from torchviz import make_dot
    # g = make_dot(out)
    # g.render('modelviz', view=False)  # 这种方式会生成一个pdf文件
    # MyConvNetVis = make_dot(out, params=dict(list(net.named_parameters()) + [('x', params[0])]))
    # MyConvNetVis.format = "pdf"
    # # 指定文件生成的文件夹
    # MyConvNetVis.directory = "/tmp/data2"
    # # 生成文件
    # MyConvNetVis.view()
    # torch.save(net, "/tmp/vis.pt")
    import netron

    def tensor_tile(tensor, repeats):
        """
        模拟tile()函数，使用torch.cat()和torch.repeat_interleave()。

        参数:
            tensor (torch.Tensor): 要进行重复操作的张量
            repeats (list or tuple): 指示每个维度上的重复次数

        返回:
            torch.Tensor: 重复后的张量
        """

        # 根据参数repeats扩展张量
        # expanded_tensor = torch.repeat_interleave(tensor, repeats[0], dim=0)

        # 如果有多个维度需要重复，则迭代扩展剩余维度
        for dim in range(0, len(repeats)):
            # if repeats[dim] != 1:
            tensor = torch.repeat_interleave(tensor, repeats[dim], dim)
        return tensor

    # def tensor_tile(tensor, repeats):

    torch.Tensor.tile = tensor_tile
    __import__("torch.onnx")
    onnx_path = "/tmp/vis2.onnx"
    torch.onnx.export(net, tuple(params), onnx_path)
    netron.start(onnx_path)

# ----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion


@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            3,
            4,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        dropout=0.10,  # List of resolutions with self-attention.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=model_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
        )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VPPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        beta_d=19.9,  # Extent of the noise level schedule.
        beta_min=0.1,  # Initial slope of the noise level schedule.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        epsilon_t=1e-5,  # Minimum t-value used during training.
        model_type="SongUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.02,  # Minimum supported noise level.
        sigma_max=100,  # Maximum supported noise level.
        model_type="SongUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".


@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        C_1=0.001,  # Timestep adjustment at low noise levels.
        C_2=0.008,  # Timestep adjustment at high noise levels.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels * 2,
            label_dim=label_dim,
            **model_kwargs,
        )

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1)
                / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1)
                - 1
            ).sqrt()
        self.register_buffer("u", u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (
            self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        )

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, : self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1),
            self.u.reshape(1, -1, 1),
        ).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        # analysis c_noise
        # boxx.cf.sigmas = boxx.cf.get("sigmas",[]) + [boxx.npa(sigma)]
        # if boxx.increase("sigma") > 5:
        #     boxx.g()
        """
loga-np.concatenate(boxx.cf.sigmas)
shape:(1005, 1, 1, 1) type:(float32 of numpy.ndarray) max: 20.767, min: 0.0062506, mean: 0.61416

loga-np.concatenate(boxx.cf.c_noise)
shape:(1005, 1, 1, 1) type:(float32 of numpy.ndarray) max: 0.57047, min: -1.1851, mean: -0.30976

show-net.model.map_noise(torch.linspace(-1., 1, 10))
net.model.map_noise(p/torch.linspace(-1, 1, 3))
[[0.54, 1.0, -0.84, -10e-05],
 [1.0, 1.0, 0.0, 0.0],
 [0.54, 1.0, 0.84, 10e-05]]
"""
        # --fp16
        # └── /: tuple 4
        #     ├── 0: (3, 3, 32, 32) of torch.cuda.HalfTensor @ cuda:0
        #     ├── 1: (3,) of torch.cuda.FloatTensor @ cuda:0
        #     ├── 2: None
        #     └── 3: dict  0
        # tree-F_x
        # └── /: (3, 3, 32, 32) of torch.cuda.HalfTensor @ cuda:0
        # tree-getpara(self.model)
        # └── /: (4, 9) of torch.cuda.FloatTensor @ cuda:0
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        # boxx.cf.debug and boxx.g()
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------


@persistence.persistent_class
class DDNPrecond(EDMPrecond):
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
