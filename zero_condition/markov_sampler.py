#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:17:54 2023

@author: yanglei
"""
from boxx.ylth import *

"""
这是我的毕业设计的核心算法, 非常非常重要, 一定要多加小心, 不许出错:
有一个 seqs, 是 torch 二维矩阵数据, seqs 的 shape 是 n, l, 表示 seq 的条数n和长度l
每一个数值都是 [0, k), k 表示 k 个状态. k = seqs.max() + 1
1. 根据 seqs 建模一条一阶马尔科夫链, 也就是 l-1 个 (k, k) 的状态转移矩阵. 同时也记录下 l[0] 的分布. 不要有 nan 和 inf, 用 0 来代替
2. 提供采样方法, 用来在马尔科夫链上采样新的 seq
3. 多使用向量操作, 而不是 for 循环
4. 用以下面向对象的代码形式

class MarkovSampler:
    def __init__(self, seqs):
        ...
    def sample(self):
        ...
"""


import torch


class MarkovSampler:
    def __init__(self, seqs):
        if isinstance(seqs, str) and seqs.endswith(".pt"):
            seqs = torch.load(seqs)
        self.seqs = seqs.long()
        self.k = seqs.max().item() + 1
        self.n, self.l = seqs.size()
        self.transition_matrices = self._build_transition_matrices()
        self.initial_distribution = self._build_initial_distribution()

    def _build_transition_matrices(self):
        transition_matrices = torch.zeros(self.l - 1, self.k, self.k)
        for i in range(self.l - 1):
            counts = torch.bincount(
                self.seqs[:, i] * self.k + self.seqs[:, i + 1],
                minlength=self.k * self.k,
            )
            transition_matrices[i] = counts.view(self.k, self.k).float()
            row_sums = transition_matrices[i].sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1
            transition_matrices[i] /= row_sums
        return transition_matrices

    def _build_initial_distribution(self):
        initial_distribution = torch.bincount(self.seqs[:, 0], minlength=self.k).float()
        initial_distribution /= initial_distribution.sum()
        return initial_distribution

    def sample(self, seed=None):
        if seed is None:
            rnd_generator = None
        else:
            rnd_generator = torch.Generator().manual_seed(seed)
        seq = -torch.ones(self.l, dtype=torch.long)
        seq[0] = torch.multinomial(
            self.initial_distribution, 1, generator=rnd_generator
        )
        for i in range(self.l - 1):
            seq[i + 1] = torch.multinomial(
                self.transition_matrices[i, seq[i]], 1, generator=rnd_generator
            )
            while not self.transition_matrices[i, seq[i], seq[i + 1]]:
                print(
                    "sample prob 0 BUG!!!\ni/sample/prob:",
                    i,
                    seq[i + 1],
                    self.transition_matrices[i, seq[i], seq[i + 1]],
                )
                seq[i + 1] = torch.multinomial(
                    self.transition_matrices[i, seq[i]], 1, generator=rnd_generator
                )
        return seq


if __name__ == "__main__":
    seqs = torch.load("../../asset/ddn_latents_l63_n6.pt")
    # seqs = torch.load("../../asset/sampler.train-ddn_latents_l63_n50000.pt")
    seqs = torch.load("../../asset/sampler.train-ddn_latents_l127_n50000.pt")
    sampler = self = MarkovSampler(seqs)
    print(seqs)
    print(sampler.sample(104))

    plot(sampler.initial_distribution, 1)
    plot(sampler.transition_matrices[20][0], 1)
