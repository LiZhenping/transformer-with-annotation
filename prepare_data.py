import os
import torch
import numpy as np
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from parsers import args
from utils import seq_padding, subsequent_mask
#数据的准备

class PrepareData:
    def __init__(self):
        # 读取数据 并分词
        #用于训练的
        self.train_en, self.train_cn = self.load_data(args.train_file)
        self.dev_en, self.dev_cn = self.load_data(args.dev_file)
        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # id化 利用上一个步的单词表，将每句话转化成数字
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
        ## 划分batch + padding + mask
        #其中重点理解mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, args.batch_size)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, args.batch_size)
        #
    def load_data(self, path):

        en = []
        cn = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.strip().split('\t')
                #分词录入，加上分隔符
                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])
        return en, cn

    def build_dict(self, sentences, max_words=50000):
        # 构造词典输入的是输入的sentences
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1
        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = args.UNK
        word_dict['PAD'] = args.PAD
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict
    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        length = len(en)
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
        return out_en_ids, out_cn_ids
    def splitBatch(self, en, cn, batch_size, shuffle=True):
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            #此处将idx_list打散用于防止过拟合，在后续从该语句中选择en cn对应的单词，个人感觉这样会有问题，因为训练语句并非是一对一的，应该训练结果不怎么样
            #具体工业上的做法待了解
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))
        return batches
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        print(args.device)
        src = torch.from_numpy(src).to(args.device).long()
        trg = torch.from_numpy(trg).to(args.device).long()

        self.src = src
        print("")
        print("------------")
        #此处unsqueeze(-2)的具体原因待了解，在倒数第二个维度上增加一个维度
        self.src_mask = (src != pad).unsqueeze(-2)
        #在该部分进行mask操作
        if trg is not None:
            #取出来除了最后一个结束符之外的所有句子，结束符不打mask
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.ntokens = (self.trg_y != pad).data.sum()
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)

    @staticmethod
    def make_std_mask(tgt, pad):
        #此处增加维度的原因待了解
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask



