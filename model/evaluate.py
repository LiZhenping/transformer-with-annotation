import time
import torch
from parsers import args
from torch.autograd import Variable
from utils import subsequent_mask
import numpy as np

def log(data, timestamp):
    file = open(f'log/log-{timestamp}.txt', 'a')
    file.write(data)
    file.write('\n')
    file.close()
#理解gready search https://blog.csdn.net/weixin_42615068/article/details/93767781 ，该步骤每步解码后，找出最大的概率再找下一个
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        #c此处利用generate找出最大的概率，将概率列表生成指定的一个单词
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
def evaluate(data, model):
    timestamp = time.time()
    with torch.no_grad():
        for i in range(len(data.dev_en)):
            #将dev_en将星数字编码
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)
            #写入log记录时间
            log(en_sent, timestamp)
            # 将dev_cn数字编码转化成单词
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))
            log(cn_sent, timestamp)
            # 转化为pytorch用的数据
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(args.device)
            src = src.unsqueeze(0)
            src_mask = (src != 0).unsqueeze(-2)

            out = greedy_decode(model, src, src_mask, max_len=args.max_length, start_symbol=data.cn_word_dict["BOS"])

            translation = []
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()]
                if sym != 'EOS':
                    translation.append(sym)
                else:
                    break
                print("translation: %s" % " ".join(translation))
                log("translation: " + " ".join(translation) + "\n", timestamp)