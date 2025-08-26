#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:37:03 2023

@author: yanglei
"""

import os
import boxx
from boxx import *

with inpkg():
    from .main import *


prompts = """man wearing sunglasses
woman wearing sunglasses
man wearing black hat
woman wearing red hat
smile boy
happy girl
old man
old lady
blonde
bearded man
handsome guy
beautiful young beauty
man wearing glasses
woman wearing glasses
American young man
American young woman
Chinese young man
Chinese young woman
Africa young man
Africa young woman"""

prompts = prompts.split("\n")


if __name__ == "__main__":
    img_glob = "~/dataset/celebA-HQ/test-256x256/*g"
    # pklp = "../../asset/v19-00004-ffhq-64x64-outputk512_leak.choice0-shot-027597.pkl"
    pklp = "../../asset/v15-00018-ffhq-64x64-blockn64_outputk512_chain.dropout0.05-shot-117913.pkl"
    outputdir = "../../asset/figure/zero_condition_appendix_v15_mask.clip"
    # pklp = "../../asset/v20-00001-ffhq-64x64-ffhq_outputk512-057702.pkl"
    # outputdir = "../../asset/figure/zero_condition_appendix_final_v20x1"
    condidat_num = 100
    target_shape = 64, 64
    if debug:
        condidat_num = 1
        outputdir = "../../junk/zero_condition/v19/"
        pklp = "../../asset/v15-00023-ffhq-64x64-blockn64_outputk64_chain.dropout0.05-shot-117913.pkl"

    outputdir = boxx.relfile(outputdir)
    net = sys._getframe(3 if boxx.sysi.gui else 0).f_globals.get("net")
    if net is None:
        print("load net....")
        net = load_net(pklp)

    imgps = sorted(glob(os.path.expanduser(img_glob)))

    imgp2idx = dict(zip(imgps, range(100000)))
    imgp2prompts = dict(zip(imgps, (prompts * 1000)))

    imgps = __import__("brainpp_yl").split_keys_by_replica(imgps)
    os.makedirs(outputdir, exist_ok=True)
    for imgp in imgps[:]:
        fname = filename(imgp)
        if debug and fname not in [
            "00144",
            "00002",
            "00136",
            "00149",
        ]:
            continue
        else:
            print(f"{increase('gen')}/{len(imgps)}")
        img = boxx.imread(imgp)
        target = uint8_to_tensor(img)
        target = nn.functional.interpolate(target[None], target_shape, mode="area")[0]

        prompt = prompts[imgp2idx[imgp] % len(prompts)]
        samplers = {
            "01reconstruction": lambda: L2Sampler(target),
            "02sr.f2": lambda: SuperResSampler(target, 1 / 2),
            "03sr.f4": lambda: SuperResSampler(target, 1 / 4),
            "04sr.f8": lambda: SuperResSampler(target, 1 / 8),
            "05sr.f16": lambda: SuperResSampler(target, 1 / 16),
            "06denoise": lambda: NoiseSampler(target),
            "07lowbit": lambda: LowBitSampler(target),
            "08color": lambda: ColorfulSampler(target),
            "09inpainting.right": lambda: L2MaskedSampler(target, 0),
            "10inpainting.down": lambda: L2MaskedSampler(target, 1),
            "11inpainting.masks": lambda: L2MaskedSampler(target, 8),
            "12inpainting.i.masks": lambda: L2MaskedSampler(target, 7),
            "13style": lambda: StyleTransfer(
                target,
            ),
            "14face.recon": lambda: FaceRecognizeSampler(target),
            "15clip": lambda: CLIPSampler(prompt),
            "16clip.wearing_sunglasses.mask9": lambda: {
                L2MaskedSampler(target, 9): 0.5,
                CLIPSampler("wearing sunglasses"): 0.5,
            },
            "17clip.wearing_hat.mask2": lambda: {
                L2MaskedSampler(target, 2): 0.5,
                CLIPSampler("wearing hat"): 0.5,
            },
            "18clip.smiling_person.mask1": lambda: {
                L2MaskedSampler(target, 1): 0.5,
                CLIPSampler("smiling person"): 0.5,
            },
        }
        for sampler_name in samplers:
            sampler = samplers[sampler_name]()
            if isinstance(sampler, dict):
                batch_sampler = MultiGuidedSampler(sampler)
            else:
                batch_sampler = BatchedGuidedSampler(sampler)
            for condidat_idx in range(condidat_num):
                d = dict(sampler=batch_sampler)
                d = net(d)
                res = t2rgb(npa(d["predict"][0]))
                target_path = pathjoin(
                    outputdir, f"{fname}_{sampler_name}_v15_{condidat_idx+1:03}.png"
                )
                if "15clip" in sampler_name:
                    target_path = target_path.replace(
                        fname, "clip-" + prompt.replace(" ", "_")
                    )
                boxx.imsave(target_path, res)
                if sampler_name == "01reconstruction":
                    break
            if boxx.sysi.gui:
                print(sampler_name)
                showd(d, 1, figsize=(8, 5) if target.shape[-1] >= 40 else (4, 3))
            if "condition0" in d and not isinstance(d["condition0"], dict):
                if d["condition0"][0].ndim == 4:
                    target_path = pathjoin(
                        outputdir, f"{fname}_{sampler_name}_{0:03}.png"
                    )
                    target_ = d["condition0"][0][0]
                    if target_.shape[-1] != target_shape[-1]:
                        target_ = nn.functional.interpolate(
                            target_[None], target_shape, mode="nearest"
                        )[0]
                    res = t2rgb(npa(target_))
                    boxx.imsave(target_path, res.squeeze())
            # break

        # show(target, t2rgb)
        # show(img)
        # break
    print("All save to:", target_path)
