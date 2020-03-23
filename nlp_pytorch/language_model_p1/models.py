# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 下午8:17
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : models.py
# @Desc    : 语言模型训练的网络结构

import torch
import torch.nn as nn


class LSTMLM(nn.Module):
    """语言模型网络架构

    Args:
        n_words (int): 词表中的单词数目
        n_embed (int): 词向量维度
        n_hidden (int): LSTM隐含状态的维度
        dropout (float): Dropout概率
        rnn_type (str): RNN类型'LSTM'或'GRU'
    """
    def __init__(self, n_words, n_embed=200, n_hidden=200, dropout=0.5, rnn_type='LSTM'):
        super(LSTMLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(n_words, n_embed)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(n_embed, n_hidden, batch_first=True)
        else:
            raise ValueError("rnn_type must in ['LSTM', 'GRU']")
        self.linear = nn.Linear(n_hidden, n_words)

    def forward(self, inputs, lengths):
        # inputs shape: (batch_size, max_length)
        # x_emb shape: (batch_size, max_length, embed_size)
        x_emb = self.drop(self.embed(inputs))

        packed_emb = nn.utils.rnn.pack_padded_sequence(x_emb, lengths, batch_first=True)
        # 这里LSTM的h_0,c_0使用全0的默认初始化，LSTM层经过后丢弃
        packed_out, _ = self.rnn(packed_emb)
        # x_out shape: (batch_size, max_length, hidden_size)
        x_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # outputs shape: (batch, max_length, vocab_size)
        return self.linear(self.drop(x_out))


class MaskCrossEntropyLoss(nn.Module):
    """含有Mask的交叉熵损失"""
    def __init__(self):
        super(MaskCrossEntropyLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets, mask):
        # outputs shape: (batch_size * max_len, vocab_size)
        outputs = outputs.view(-1, outputs.size(2))
        # targets shape: (batch_size * max_len)
        targets = targets.view(-1)
        # mask shape: (batch_size * max_len)
        mask = mask.view(-1)
        loss = self.celoss(outputs, targets) * mask
        return torch.sum(loss) / torch.sum(mask)
