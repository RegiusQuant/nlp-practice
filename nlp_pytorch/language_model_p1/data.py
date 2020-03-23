# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 下午8:03
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : data.py
# @Desc    : 语言模型所需的数据定义

from pathlib import Path

import torch


class Vocab:
    """单词表类

    Args:
        vocab_path (Path): 单词表文件路径

    Attributes:
        stoi (Dict): 单词->索引 字典
        itos (List): 索引->单词 列表
    """
    def __init__(self, vocab_path: Path):
        self.stoi = {}  # token -> index (dict)
        self.itos = []  # index -> token (list)

        with open(vocab_path) as f:
            # bobsue.voc.txt中，每一行是一个单词
            for w in f.readlines():
                w = w.strip()
                if w not in self.stoi:
                    self.stoi[w] = len(self.itos)
                    self.itos.append(w)

    def __len__(self):
        return len(self.itos)


class Corpus:
    """语料库类
    
    Args:
        data_path (Path): 语料库所在文件夹路径
        sort_by_len (bool): 语料是否按照长度降序排列

    Attributes:
        vocab (Vocab): 单词表实例
        train_data (List): 训练数据, 已经处理为单词索引编号列表
        valid_data (List): 验证数据
        test_data (List): 测试数据
    """
    def __init__(self, data_path: Path, sort_by_len: bool = False):
        self.vocab = Vocab(data_path / 'bobsue.voc.txt')
        self.sort_by_len = sort_by_len
        self.train_data = self.tokenize(data_path / 'bobsue.lm.train.txt')
        self.valid_data = self.tokenize(data_path / 'bobsue.lm.dev.txt')
        self.test_data = self.tokenize(data_path / 'bobsue.lm.test.txt')

    def tokenize(self, text_path: Path):
        with open(text_path) as f:
            index_data = []  # 索引数据，存储每个样本的单词索引列表
            for s in f.readlines():
                index_data.append(self.sentence_to_index(s))
        if self.sort_by_len:  # 为了提升训练速度，可以考虑将样本按照长度排序，这样可以减少padding
            index_data = sorted(index_data, key=lambda x: len(x), reverse=True)
        return index_data

    def sentence_to_index(self, s):
        return [self.vocab.stoi[w] for w in s.split()]

    def index_to_sentence(self, x):
        return ' '.join([self.vocab.itos[i] for i in x])


class BobSueLMDataSet(torch.utils.data.Dataset):
    """语言模型数据集"""
    def __init__(self, index_data):
        self.index_data = index_data

    def __getitem__(self, i):
        # 根据语言模型定义，这里我们要用前n-1个单词预测后n-1个单词
        return self.index_data[i][:-1], self.index_data[i][1:]

    def __len__(self):
        return len(self.index_data)
