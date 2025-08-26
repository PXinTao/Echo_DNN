#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 00:25:45 2023

@author: yanglei

摘抄至 ipynb  v16 ffhq condition 256
"""

# v16-00009-ffhq-256x256-ffhq256_cond.edge_chain.dropout0.05-shot-055193.pkl
gennum = 20

dirr = "../asset/figure/condition_ddn/"
os.makedirs(dirr, exist_ok=True)
for geni in range(gennum):
    with torch.no_grad():
        d = net(dc_celeba.copy())
    #     showd(d, 1)
    for batchi, predict in enumerate(d["predict"]):
        rgb = t2rgb(predict)
        imsave(dirr + f"edge.condition{seleted_idxs[batchi]:05}.gen{geni:03}.png", rgb)
        if geni == 0:
            condition = t2rgb(d["condition"][batchi]).squeeze()
            imsave(
                dirr + f"edge.condition{seleted_idxs[batchi]:05}.condition.png",
                condition,
            )

showd(net(d_bad_target.copy()), 1)

# rm ../asset/figure/condition_ddn/color*.png


for geni in range(len(seleted_idxs)):
    with torch.no_grad():
        d_bad_target = dc_celeba.copy()
        d_bad_target["target"] = d_celeba["target"][[geni] * len(seleted_idxs)]
        d = net(d_bad_target.copy())
    #     showd(d, 1)
    zero = t2rgb(d_bad_target["target"][0])
    imsave(dirr + f"edge.guided_by{seleted_idxs[geni]:05}.0.png", zero)

    for batchi, predict in enumerate(d["predict"]):
        rgb = t2rgb(predict)
        imsave(
            dirr
            + f"edge.guided_by{seleted_idxs[geni]:05}.condition{seleted_idxs[batchi]:05}.png",
            rgb,
        )


showd(net(d_bad_target.copy()), 1)
