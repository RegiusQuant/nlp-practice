# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 上午11:52
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : data.py
# @Desc    : 语言模型所需的数据存储结构

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class Vocab:
    """单词表类, 用于存储单词相关的数据结构
    
    Args:
        vocab_path (Path): 单词表文件路径
       
    Attributes:
        stoi (Dict): 存储每个单词对应索引的字典
        itos (List): 单词列表
         
    """
    def __init__(self, vocab_path: Path):
        self.stoi = {}  # token -> index (dict)
        self.itos = []  # index -> token (list)

        with open(vocab_path) as f:
            # bobsue.voc.txt中, 每一行是一个单词
            for w in f.readlines():
                w = w.strip()
                if w not in self.stoi:
                    self.stoi[w] = len(self.itos)
                    self.itos.append(w)

    def __len__(self):
        return len(self.itos)


class SentCorpus:
    """基于单句训练的语料库
    
    Args:
        data_path (Path): 数据集路径
        uniform (bool): 是否采用均匀分布采样
        freq_coef (float): 使用词频采样时的提升系数
        
    Attributes:
        vocab (Vocab): 单词表类的实例
        train_data (List): 保存训练句子单词索引的列表
        valid_data (List): 保存验证句子单词索引的列表
        test_data (List): 保存测试句子单词索引的列表
        word_counter (Counter): 保存有训练句子中出现单词的计数器
        word_freqs (List): 训练数据中的单词采样词频
    """
    def __init__(self, data_path: Path, uniform: bool = False, freq_coef: float = 0.1):
        self.vocab = Vocab(data_path / 'bobsue.voc.txt')
        self.train_data = self.tokenize(data_path / 'bobsue.lm.train.txt')
        self.valid_data = self.tokenize(data_path / 'bobsue.lm.dev.txt')
        self.test_data = self.tokenize(data_path / 'bobsue.lm.test.txt')

        # 统计训练集的单词计数
        self.word_counter = Counter()
        for x in self.train_data:
            # 注意<s>不在我们的预测范围内，不要统计
            self.word_counter += Counter(x[1:])

        if uniform:  # 均匀分布
            self.word_freqs = np.array([0.] + [1. for _ in range(len(self.vocab) - 1)], dtype=np.float32)
            self.word_freqs = self.word_freqs / sum(self.word_freqs)
        else:  # 词频分布（提升freq_coef次方）
            self.word_freqs = np.array([self.word_counter[i] for i in range(len(self.vocab))], dtype=np.float32)
            self.word_freqs = self.word_freqs / sum(self.word_freqs)
            self.word_freqs = self.word_freqs**freq_coef
            self.word_freqs = self.word_freqs / sum(self.word_freqs)

    def tokenize(self, text_path: Path) -> List[List[int]]:
        """将文本中所有句子转换为单词序号列表
        
        Args:
            text_path (Path): 文本路径 

        Returns:
            index_data (List): 处理为序号的列表
        """
        with open(text_path) as f:
            index_data = []  # 索引数据，存储每个样本的单词索引列表
            for s in f.readlines():
                index_data.append(self.sentence_to_index(s))
        return index_data

    def sentence_to_index(self, s: str) -> List[int]:
        """将由字符串表示的一句话转换为单词序号列表
        
        Args:
            s (str): 句子字符串

        Returns:
            result (List): 句子转换后的序号列表
        """
        return [self.vocab.stoi[w] for w in s.split()]

    def index_to_sentence(self, x: List[int]) -> str:
        """将单词序号列表转换成对应的字符串
        
        Args:
            x (List): 由单词序号构成的列表

        Returns:
            result (str): 序号列表转换后的字符串
        """
        return ' '.join([self.vocab.itos[i] for i in x])


class NegSampleDataSet(torch.utils.data.Dataset):
    """负例采样的PyTorch数据集
    
    Args:
        index_data (List): 语料库中的编号数据
        word_freqs (List): 负例采样的词频列表
        n_negs (int): 负例采样数目
    """
    def __init__(self, index_data: List[List[int]], word_freqs: List[float], n_negs: int = 20):
        self.index_data = index_data  # 转换为序号的文本
        self.n_negs = n_negs  # 生成负例个数
        self.word_freqs = torch.FloatTensor(word_freqs)  # 词频

    def __getitem__(self, i):
        inputs = torch.LongTensor(self.index_data[i][:-1])
        poss = torch.LongTensor(self.index_data[i][1:])

        # 生成n_negs个负例
        negs = torch.zeros((len(poss), self.n_negs), dtype=torch.long)
        for i in range(len(poss)):
            negs[i] = torch.multinomial(self.word_freqs, self.n_negs)

        return inputs, poss, negs

    def __len__(self):
        return len(self.index_data)


def neglm_collate_fn(batch):
    # 首先将batch的格式进行转换
    # batch[0]：Inputs
    # batch[1]: Poss
    # batch[2]: Negs
    batch = list(zip(*batch))

    # lengths: (batch_size)
    lengths = torch.LongTensor([len(x) for x in batch[0]])
    # inputs: (batch_size, max_len)
    inputs = nn.utils.rnn.pad_sequence(batch[0], batch_first=True)
    # poss: (batch_size, max_len)
    poss = nn.utils.rnn.pad_sequence(batch[1], batch_first=True)
    # negs: (batch_size, max_len, n_negs)
    negs = nn.utils.rnn.pad_sequence(batch[2], batch_first=True)
    # mask: (batch_size, max_len)
    mask = (poss != 0).float()

    return inputs, poss, negs, lengths, mask


class ContextCorpus:
    """基于上下文的语料库
    
    Args:
        data_path (Path): 数据集路径
        
    Attributes:
        vocab (Vocab): 单词表类的实例
        train_data (List): 保存训练上下文句子单词索引的列表
        valid_data (List): 保存验证上下文句子单词索引的列表
        test_data (List): 保存测试上下文句子单词索引的列表
    """
    def __init__(self, data_path: Path):
        self.vocab = Vocab(data_path / 'bobsue.voc.txt')
        self.train_data = self.tokenize(data_path / 'bobsue.prevsent.train.tsv')
        self.valid_data = self.tokenize(data_path / 'bobsue.prevsent.dev.tsv')
        self.test_data = self.tokenize(data_path / 'bobsue.prevsent.test.tsv')

    def tokenize(self, text_path: Path) -> List[Tuple[List[int], List[int]]]:
        """将文本中上下文句子转换为单词序号列表

        Args:
            text_path (Path): 文本路径

        Returns:
            index_data (List): 经过处理后的列表
        """
        with open(text_path) as f:
            index_data = []
            for s in f.readlines():
                t = s.split('\t')
                index_data.append((self.sentence_to_index(t[0]), self.sentence_to_index(t[1])))
            return index_data

    def sentence_to_index(self, s):
        return [self.vocab.stoi[w] for w in s.split()]

    def index_to_sentence(self, x):
        return ' '.join([self.vocab.itos[i] for i in x])


class ContextDataset(torch.utils.data.Dataset):
    """基于上下文的PyTorch数据集

    Args:
        index_data (List): 语料库中的编号数据

    """
    def __init__(self, index_data: List[Tuple[List[int], List[int]]]):
        self.index_data = index_data

    def __getitem__(self, i):
        contexts = torch.LongTensor(self.index_data[i][0])
        inputs = torch.LongTensor(self.index_data[i][1][:-1])
        targets = torch.LongTensor(self.index_data[i][1][1:])
        return contexts, inputs, targets

    def __len__(self):
        return len(self.index_data)


def ctxlm_collate_fn(batch):
    # 首先将batch的格式进行转换
    # batch[0]：Contexts
    # batch[1]: Inputs
    # batch[2]: Targets
    batch = list(zip(*batch))

    ctx_lengths = torch.LongTensor([len(x) for x in batch[0]])
    inp_lengths = torch.LongTensor([len(x) for x in batch[1]])

    contexts = nn.utils.rnn.pad_sequence(batch[0], batch_first=True)
    inputs = nn.utils.rnn.pad_sequence(batch[1], batch_first=True)
    targets = nn.utils.rnn.pad_sequence(batch[2], batch_first=True)

    mask = (targets != 0).float()

    return contexts, inputs, targets, ctx_lengths, inp_lengths, mask