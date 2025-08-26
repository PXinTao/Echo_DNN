#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 00:07:50 2023

@author: yanglei
"""

import os
import boxx
from boxx import *

with inpkg():
    from .main import *


from ddn_utils import *

net = loadnet(
    "https://oss.iap.hh-d.brainpp.cn/yl-project/ddn/exps/v16_condition/00016-cifar10-32x32-cifar_fp32_cond.class_blockn32_outputk512_chain.dropout0.05_transfer.00015.kimg.175616/shot-005018.pkl"
)

dirr = "../asset/figure/condition_cifar"

gen_dirr = dirr + "/pngs"
os.makedirs(gen_dirr, exist_ok=True)

imgps = []
for c in range(10):
    label = torch.zeros(100, 10).cuda()
    label[:, c] = 1
    d = net(dict(batch_size=len(label), class_labels=label))
    for bi, pre in enumerate(d["predict"]):
        res = t2rgb(pre)
        imgp = f"{gen_dirr}/{bi*10+c:04}.png"
        imsave(imgp, res)
        imgps.append(imgp)

make_vis_img(sorted(imgps)[:100], dirr + "/vis.png")
showd(d, 1)
