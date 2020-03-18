# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 下午12:19
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : main.py
# @Desc    : 语言模型主文件

import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data import SentCorpus, ContextCorpus
from learner import NegSampleLearner, ContextLearner


def set_random_seed(seed):
    """设定随机种子, 保证试验结果稳定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show_device_info():
    """显示设备信息"""
    print('PyTorch Version:', torch.__version__)
    print('-' * 60)
    if torch.cuda.is_available():
        print('CUDA Device Count:', torch.cuda.device_count())
        print('CUDA Device Name:')
        for i in range(torch.cuda.device_count()):
            print('\t', torch.cuda.get_device_name(i))
        print('CUDA Current Device Index:', torch.cuda.current_device())
        print('-' * 60)


def run_homework1_q1():
    """运行基于负例采样的语言模型"""
    set_random_seed(2020)
    show_device_info()

    data_path = Path('/media/bnu/data/nlp-practice/language-model')
    corpus = SentCorpus(data_path, uniform=False, freq_coef=0.01)
    learner = NegSampleLearner(corpus, n_embed=200, dropout=0.5, n_negs=20, batch_size=128)
    learner.fit(20)
    print('Max Accuarcy in Valid', max(learner.history['valid_acc']))
    test_acc, total_word = learner.predict()
    print('Test Accuracy:', test_acc)


def run_homework1_q2():
    """运行基于上下文信息的语言模型"""
    set_random_seed(2020)
    show_device_info()

    data_path = Path('/media/bnu/data/nlp-practice/language-model')
    corpus = ContextCorpus(data_path)
    learner = ContextLearner(corpus, n_embed=700, n_hidden=700, dropout=0.5)
    learner.fit(1000)
    print('Accuracy in Validation:', max(learner.history['valid_acc']))

    test_loss, test_acc, test_words, test_result = learner.predict()
    print('测试集上的结果 --> Loss: {:.3f}, Acc: {:.3f}, Words: {}'.format(
        test_loss, test_acc, test_words))

    mistake_counter = Counter()
    for i in range(len(test_result['targets'])):
        for j in range(len(test_result['targets'][i])):
            pred, target = test_result['preds'][i][j], test_result['targets'][i][j]
            if pred != target:
                pred, target = corpus.vocab.itos[pred], corpus.vocab.itos[target]
                mistake_counter[(target, pred)] += 1
    for k, v in mistake_counter.most_common(35):
        print(k, v)


if __name__ == '__main__':
    run_homework1_q1()
    run_homework1_q2()
