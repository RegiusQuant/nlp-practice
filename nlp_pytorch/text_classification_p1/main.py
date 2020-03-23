# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 下午3:10
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : main.py
# @Desc    : 情感分析主逻辑

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import tqdm

from data import SSTCorpus
from models import EmbedAvgModel, AttnAvgModel
from learner import TextClassificationLearner


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show_device_info():
    print('PyTorch Version:', torch.__version__)
    print('-' * 60)
    if torch.cuda.is_available():
        print('CUDA Device Count:', torch.cuda.device_count())
        print('CUDA Device Name:')
        for i in range(torch.cuda.device_count()):
            print('\t', torch.cuda.get_device_name(i))
        print('CUDA Current Device Index:', torch.cuda.current_device())
        print('-' * 60)


def run_part1_q1():
    set_random_seed(2020)
    show_device_info()

    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    corpus = SSTCorpus(data_path)
    train_iter, valid_iter, test_iter = corpus.get_iterators(batch_sizes=(256, 256, 256))

    ############################################################
    # 模型训练
    ############################################################
    model = EmbedAvgModel(n_words=len(corpus.text_field.vocab), n_embed=200, p_drop=0.5,
                          padding_idx=corpus.get_padding_idx())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = TextClassificationLearner(model, optimizer, corpus.get_padding_idx(),
                                        save_path='./models/embed-avg-best.pth')
    learner.fit(train_iter, valid_iter, n_epochs=20)
    learner.predict(test_iter)

    ############################################################
    # 单词L2-Norm分析
    ############################################################
    # (n_words)
    embed_norm = model.embed.weight.norm(dim=1)
    word_idx = list(range(len(corpus.text_field.vocab)))
    word_idx.sort(key=lambda x: embed_norm[x])
    print('\n')
    print('15个L2-Norm最小的单词：')
    for i in word_idx[:15]:
        print(corpus.text_field.vocab.itos[i])
    print('-' * 60)
    print('15个L2-Norm最大的单词：')
    for i in word_idx[-15:]:
        print(corpus.text_field.vocab.itos[i])


def run_part2_q2():
    set_random_seed(2020)
    show_device_info()

    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    corpus = SSTCorpus(data_path)
    train_iter, valid_iter, test_iter = corpus.get_iterators(batch_sizes=(256, 256, 256))

    model = AttnAvgModel(n_words=len(corpus.text_field.vocab), n_embed=150, p_drop=0.5,
                         padding_idx=corpus.get_padding_idx())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = TextClassificationLearner(model, optimizer, corpus.get_padding_idx(),
                                        save_path='./models/attn-avg-best.pth')
    # learner.fit(train_iter, valid_iter, n_epochs=50)
    learner.predict(test_iter)

    ############################################################
    # 余弦相似度分析
    ############################################################
    # (1, n_embed)
    u = model.coef.view(1, -1)
    # (n_words, n_embed)
    embedding = model.embed.weight
    # (n_words)
    cos_sim = F.cosine_similarity(u, embedding, dim=-1)

    word_idx = list(range(len(corpus.text_field.vocab)))
    word_idx.sort(key=lambda x: cos_sim[x])

    print('15个余弦相似度最小的单词：')
    for i in word_idx[:15]:
        print(corpus.text_field.vocab.itos[i])
    print('-' * 60)

    print('15个余弦相似度最大的单词：')
    for i in word_idx[-15:]:
        print(corpus.text_field.vocab.itos[i])

    ############################################################
    # 分析单词Attention权重
    ############################################################
    train_iter, valid_iter, test_iter = corpus.get_iterators(batch_sizes=(1, 1, 1))
    weight_dict = defaultdict(list)

    with torch.no_grad():
        for k, batch in enumerate(train_iter):
            inputs, lengths = batch.text
            attn = model.calc_attention_weight(inputs)
            inputs = inputs.view(-1)
            attn = attn.view(-1)
            if inputs.shape[0] == 1:
                weight_dict[inputs.item()].append(attn.item())
            else:
                for i in range(len(inputs)):
                    weight_dict[inputs[i].item()].append(attn[i].item())
            if (k + 1) % 10000 == 0:
                print(f'{k+1} sentences finish!')

    mean_dict, std_dict = {}, {}
    for k, v in weight_dict.items():
        # 至少出现100次
        if len(v) >= 100:
            mean_dict[k] = np.mean(v)
            std_dict[k] = np.std(v)

    word_idx = list(std_dict.keys())
    word_idx.sort(key=lambda x: std_dict[x], reverse=True)
    print('30个Attention标准差最大的单词：')
    print('-' * 60)
    for i in word_idx[:30]:
        print(f'{corpus.text_field.vocab.itos[i]}, {len(weight_dict[i])}, {std_dict[i]:.3f}, {mean_dict[i]:.3f}')
    print()
    print('30个Attention标准差最小的单词：')
    print('-' * 60)
    for i in reversed(word_idx[-30:]):
        print(f'{corpus.text_field.vocab.itos[i]}, {len(weight_dict[i])}, {std_dict[i]:.3f}, {mean_dict[i]:.3f}')


if __name__ == '__main__':
    # run_part1_q1()
    run_part2_q2()
