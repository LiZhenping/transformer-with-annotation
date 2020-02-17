# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:36:18 2020

@author: lizhenping
encoder用于搭载transformer模型的encoder的外层循环框架，此处理解循环是多头注意力的实现（multi-head attention），
同时多个layer并行的分别运算，捕获的各自的注意力。
encoderlayer用于具体实现每个layer中的具体注意力模型运算，在其中分别调用函数实现包括注意力机制，
layernorm的实现norm，在sublayer中实现残差连接
EncoderLayer中的神经网络有两层，第一层实现注意力，及残差计算，其中的主要目标是第一步计算norm，第二部将norm后的结果计算attention ，第三步进行残差运算执行add。
第二层直接进行残差运算直接进行神经网络运算。第二层没有特别的计算主要执行框架中的add
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clones
from model.sublayer import SublayerConnection, LayerNorm

class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # 连续encode 6次，且是循环的encode
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        #搭建框架中的attention层和feedforward层
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    """
    SublayerConnection的作用就是把multi和ffn连在一起，只不过每一层输出之后都要先norm再残差
    """
    def forward(self, x, mask):
        print(x)
        #分别执行搭建的attention层（sublayer[0]）和feedforward层（sublayer[1]）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)