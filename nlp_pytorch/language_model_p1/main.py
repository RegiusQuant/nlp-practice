# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 下午8:00
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : main.py
# @Desc    : 语言模型主文件

import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data import Corpus
from learner import LMLearner


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


def run(learner):
    """模型训练"""
    print(learner.model)
    print('-' * 60)
    history = learner.fit(1000)
    print('Accuracy in Validation:', max(history['valid_acc']))
    print('-' * 60)


def show_test_sample(sample_index, test_result, corpus):
    """展示结果样例"""
    print('Test Sample：')
    print('Pred Values\t', test_result['preds'][sample_index])
    print('True Values\t', test_result['targets'][sample_index])
    print('Pred Sentence\t', corpus.index_to_sentence(test_result['preds'][sample_index]))
    print('True Sentence\t', corpus.index_to_sentence(test_result['targets'][sample_index]))
    print('-' * 60)


def show_most_mistake(test_result, corpus):
    """展示常见的错误"""
    mistake_counter = Counter()
    for i in range(len(test_result['targets'])):
        for j in range(len(test_result['targets'][i])):
            pred, target = test_result['preds'][i][j], test_result['targets'][i][j]
            if pred != target:
                pred, target = corpus.vocab.itos[pred], corpus.vocab.itos[target]
                mistake_counter[(target, pred)] += 1
    for k, v in mistake_counter.most_common(35):
        print(k, v)


def main():
    set_random_seed(2020)
    show_device_info()

    data_path = Path('/media/bnu/data/nlp-practice/language-model')
    corpus = Corpus(data_path, sort_by_len=False)

    learner = LMLearner(corpus, n_embed=400, n_hidden=400, dropout=0.5,
                        rnn_type='LSTM', batch_size=128, learning_rate=1e-3)
    # 训练模型, 已经训练好进行错误分析的时候可以注释掉
    run(learner)

    test_loss, test_acc, test_words, test_result = learner.predict()
    print('Result in Test --> Loss: {:.3f}, Acc: {:.3f}, Words: {}'.format(test_loss, test_acc, test_words))

    show_test_sample(4, test_result, corpus)
    show_most_mistake(test_result, corpus)


if __name__ == '__main__':
    main()
