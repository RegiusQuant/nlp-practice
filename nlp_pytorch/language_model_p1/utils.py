# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 下午8:13
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : utils.py
# @Desc    : 语言模型工具库

import torch
import torch.nn as nn


def lm_collate_fn(batch):
    """DataLoader所需的整理函数,解决输入数据长度不一的问题"""
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # 这里输入的batch格式为[(input_1, target_1), ... ,(input_n, target_n)]
    # 我们要将其格式转换为[(input_1, ... , input_n), (target_1, ... , target_n)]
    batch = list(zip(*batch))

    # 生成长度列表
    lengths = torch.LongTensor([len(x) for x in batch[0]]).to(device)

    # 对输入和目标进行padding
    inputs = [torch.LongTensor(x).to(device) for x in batch[0]]
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = [torch.LongTensor(x).to(device) for x in batch[1]]
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

    # 因为目标中不存在编号为0的单词，所以目标中为0的位置为padding，由此生成mask矩阵
    mask = (targets != 0).float().to(device)

    # 在之后的训练中因为还要进行pack_padded_sequence操作，所以在这里按照长度降序排列
    lengths, perm_index = lengths.sort(descending=True)
    inputs = inputs[perm_index]
    targets = targets[perm_index]
    mask = mask[perm_index]

    return inputs, targets, lengths, mask
