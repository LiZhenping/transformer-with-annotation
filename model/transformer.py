# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:11:07 2020

@author: lizhenping
#此处定义transformer的框架结构，其中包含encoder结构，decoder结构，继承nn类，
# 在nn的forward中将encoder的返回值输入decoder中
# （encoder通过学习输入，将其编码成一个固定大小的状态向量s，继而将s传给decoder，decoder再通过对状态向量s的学习来进行输出。）

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    #定义encoderdecoder类，继承nn.module，同时传入encoder类，decoder类，经过embedding的待翻译语句src_embed,
    #经过embedding的已翻译语句tgt_embed
    #generator用于做softmax取得分类后的预测值
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    #在encoder中传入待训练的x(src)，x的mask规则(src_mas)
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    # 在decoder中传入目标y(tgt)，y的mask规则(tgt_mask)，x的mask规则(src_mas)，
    #其中memory
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

