import torch
import math, copy
import torch.nn.functional as F
import torch.nn as nn

from utils import clones
#此处实现attention公式，分别将K Q先进行相乘，随后得到的score再乘以V
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        #注意此处返回的是两个值，分别是训练后的attention 和 p_attn，其中第一个用于后面的函数，传入，p_attn的作用尚未得知
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        #此处执行多头注意力模型将数据从[14, 5, 256]变为[14, 5, 8, 32]
        #此处应该理解多头注意力，实际上，是把一句话拆分为多个话，拆开后，分别去捕获，句子中的关联关系。
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             #此处理解线性变化，文中使用的是nn.ModuleList()搭载神经网络，利用python字典[]语法调用 l[0](x),l[1](x),l[2](x)层进行运算，此处是简化技巧理解起来较为抽象
             #主要理解是其中的语法对q,k,v和四层神经网络的调用传值其中前三层在该处调用最后一层在返回值中调用具体语法理解见demo
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


'''
###demo
class test():
    def add(self,a):
        c=a+a

        return c

class test2():
    def add(self,a):
        c=a*100

        return c
a=test()
b=test2()
tmp_list = []
tmp_list.append(a)
tmp_list.append(b)
test1=1
test2=2


query = [l.add(x) for l, x in zip(tmp_list, (test1,test2))]
print(query)
'''