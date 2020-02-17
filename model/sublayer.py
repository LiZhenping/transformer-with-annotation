import torch
import torch.nn as nn

from parsers import args
"""
Created on Wed Feb 12 21:11:07 2020

@author: lizhenping
#在此处搭建encoder，decoder中的sublayer层，在其中实现norm和add操作
# 在sublayer中分别实现了layernorm和dropout
# （encoder通过学习输入，将其编码成一个固定大小的状态向量s，继而将s传给decoder，decoder再通过对状态向量s的学习来进行输出。）

"""

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))