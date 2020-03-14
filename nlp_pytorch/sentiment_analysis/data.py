# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 下午8:01
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : data.py
# @Desc    : 酒店评价数据处理

from pathlib import Path

import jieba
import torch
import torchtext


class HotelSentiCorp:

    def __init__(self, data_path):
        self.text_field = torchtext.data.Field(
            sequential=True,
            use_vocab=True,
            tokenize=jieba.lcut,
            batch_first=True
        )
        self.label_field = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float)

        self.raw_data = torchtext.data.TabularDataset(
            path=data_path,
            format='csv',
            skip_header=True,
            fields=[
                ('label', self.label_field),
                ('text', self.text_field)
            ]
        )
        self.train_data, self.valid_data = self.raw_data.split(split_ratio=0.8)

    def get_iterators(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_field.build_vocab(
            self.train_data,
            max_size=25000,
        )
        train_iter, valid_iter = torchtext.data.BucketIterator.splits(
            (self.train_data, self.valid_data),
            batch_sizes=(64, 64),
            sort_key=lambda x: len(x.text),
            device=device,
            shuffle=True
        )
        return train_iter, valid_iter

