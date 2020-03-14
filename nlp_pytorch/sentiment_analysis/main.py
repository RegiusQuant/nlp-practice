# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 下午9:40
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : main.py
# @Desc    : 情感分析运行逻辑

from pathlib import Path

import torch

from data import HotelSentiCorp
from models import WordAvgModel
from learner import SALearner


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    corp = HotelSentiCorp(
        data_path=Path('/media/bnu/data/nlp-practice/sentiment-analysis/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.txt'))
    print('Corpus Example Text:')
    print(corp.raw_data.examples[0].text)
    print('Corpus Example Label:')
    print(corp.raw_data.examples[0].label)
    print('-' * 100)
    print('Number of Train Data:', len(corp.train_data))
    print('Number of Valid Data:', len(corp.valid_data))
    print('-' * 100)

    train_iter, valid_iter = corp.get_iterators()
    batch = next(iter(train_iter))
    print(batch)
    print(batch.label)
    print('-' * 100)

    n_words = len(corp.text_field.vocab)
    mask = (batch.text != corp.text_field.vocab.stoi['<pad>']).float()
    print('Mask Shape:', mask.shape)
    print('-' * 100)

    model = WordAvgModel(n_words=n_words, n_embed=200, dropout=0.2)
    model.to(device)
    outputs = model(batch.text, mask)
    print('WordAVGModel Outputs Shape:', outputs.shape)

    learner = SALearner(corp, model, device)
    learner.fit(train_iter, valid_iter, 10)


if __name__ == '__main__':
    main()
