import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from parsers import args
#该方法主要用于将输入的单词进行进行embedding，为语句处理的最开始部分。注意区别同PositionalEncoding的区别，在框架图位置中该模块用于input embedding和output embedding
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
#该模块用于关联x的位置信息，通过传入Transformer中的src_embed,调用该方法，传入src，使用的调用函数是src_embed(src)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=args.device)
        #position=[0,1,....,4999]此处是简写，其实是4999.9999因为pytorch生成的数字并非是整数
        position = torch.arange(0., max_len,  device=args.device).unsqueeze(1)
        #该处在李宏毅课程讲过，手动设置位置信息不是使用深度学习学习的，但是没有具体的理论公式，只是这样尝试后，效果比较好
        div_term = torch.exp(torch.arange(0., d_model, 2,  device=args.device) *- (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 加1个维度
        #设置该参数不更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        #此处执行加上位置参数的操作
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)