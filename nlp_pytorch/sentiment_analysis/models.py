# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 下午9:29
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : models.py
# @Desc    : 情感分类模型定义

import torch
import torch.nn as nn


class WordAvgModel(nn.Module):
    def __init__(self, n_words, n_embed, dropout=0.2):
        super(WordAvgModel, self).__init__()
        self.embed = nn.Embedding(n_words, n_embed)
        self.linear = nn.Linear(n_embed, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        x_embed = self.embed(inputs)
        x_embed = self.dropout(x_embed)  # (batch_size, max_length, embedding_size)

        mask = mask.unsqueeze(2)  # (batch_size, max_length, 1)
        x_embed = x_embed * mask

        x_out = x_embed.sum(1) / (mask.sum(1) + 1e-9)  # (batch_size, embedding_size)

        return self.linear(x_out).squeeze()


