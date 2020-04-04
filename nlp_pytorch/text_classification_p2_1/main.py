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
from models import SimpleSelfAttentionModel, SimpleTransformerModel, PyTorchTransformerModel
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


def run_part2_q1():
    set_random_seed(2020)
    show_device_info()

    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    corpus = SSTCorpus(data_path)
    train_iter, valid_iter, test_iter = corpus.get_iterators(batch_sizes=(256, 256, 256))

    model = SimpleSelfAttentionModel(
        n_words=len(corpus.text_field.vocab),
        n_embed=200,
        p_drop=0.2,
        pad_idx=corpus.text_field.vocab.stoi['<pad>'],
        res_conn=False,
        score_fn='cos',
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = TextClassificationLearner(model, optimizer, corpus.get_padding_idx(),
                                        save_path='./models/self-attn-best.pth')
    learner.fit(train_iter, valid_iter, n_epochs=50)
    learner.predict(test_iter)


def run_part2_q2():
    set_random_seed(2020)
    show_device_info()

    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    corpus = SSTCorpus(data_path)
    train_iter, valid_iter, test_iter = corpus.get_iterators(batch_sizes=(256, 256, 256))

    model = SimpleTransformerModel(
        n_words=len(corpus.text_field.vocab),
        d_model=200,
        n_heads=2,
        p_drop=0.2,
        use_pos=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = TextClassificationLearner(model, optimizer, corpus.get_padding_idx(),
                                        save_path='./models/sim-trans-best.pth')
    learner.fit(train_iter, valid_iter, n_epochs=50)
    learner.predict(test_iter)


def run_part2_q3():
    set_random_seed(2020)
    show_device_info()

    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    corpus = SSTCorpus(data_path)
    train_iter, valid_iter, test_iter = corpus.get_iterators(batch_sizes=(256, 256, 256))

    model = PyTorchTransformerModel(
        n_words=len(corpus.text_field.vocab),
        d_model=200,
        n_heads=2,
        n_layers=3,
        p_drop=0.2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = TextClassificationLearner(model, optimizer, corpus.get_padding_idx(),
                                        save_path='./models/trans-best.pth')
    learner.fit(train_iter, valid_iter, n_epochs=50)
    learner.predict(test_iter)


if __name__ == '__main__':
    # run_part2_q1()
    # run_part2_q2()
    run_part2_q3()
