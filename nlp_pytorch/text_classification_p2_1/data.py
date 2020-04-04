# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 下午3:09
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : data.py
# @Desc    : 情感分析数据处理模块

from core import *


class SSTCorpus:
    """Standord-Sentiment-Treebank语料库

    Args:
        data_path (Path): 数据路径

    Attributes:
        text_field (Field): 文本处理方式
        label_field (LabelField): 标签处理方式
        train_set (TabularDataset): 训练数据
        valid_set (TabularDataset): 验证数据
        test_set (TabularDataset): 测试数据
    """
    def __init__(self, data_path: Path):
        self.data_path = data_path

        self.text_field = Field(sequential=True, batch_first=True, include_lengths=True)
        self.label_field = Field(sequential=False, use_vocab=False, dtype=torch.float)

        self.train_set, self.valid_set, self.test_set = TabularDataset.splits(
            path=str(self.data_path),
            train='senti.train.tsv',
            validation='senti.dev.tsv',
            test='senti.test.tsv',
            format='tsv',
            fields=[('text', self.text_field), ('label', self.label_field)],
        )
        self.text_field.build_vocab(self.train_set)

    def get_iterators(self, batch_sizes: Tuple) -> Tuple:
        """获取训练/验证/测试的数据迭代器
        
        Args:
            batch_sizes (Tuple): 训练/验证/测试的BatchSize

        Returns:
            train_iter (BucketIterator): 训练集数据迭代器
            valid_iter (BucketIterator): 验证集数据迭代器
            test_iter (BucketIterator): 测试集数据迭代器
        """
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            datasets=(self.train_set, self.valid_set, self.test_set),
            batch_sizes=batch_sizes,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=DEVICE,
        )
        return train_iter, valid_iter, test_iter

    def get_padding_idx(self):
        """获取<pad>的索引

        Returns:
            padding_idx (int): <pad>的索引
        """
        return self.text_field.vocab.stoi['<pad>']


############################################################
# 测试语料库
############################################################
def test_corpus():
    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    corpus = SSTCorpus(data_path)

    print('Tabular Dataset Example:')
    print('Text:', corpus.valid_set[10].text)
    print('Label:', corpus.valid_set[10].label)
    print('-' * 60)

    print('Vocab: Str -> Index')
    print(list(corpus.text_field.vocab.stoi.items())[:5])
    print('Vocab: Index -> Str')
    print(corpus.text_field.vocab.itos[:5])
    print('Vocab Size:')
    print(len(corpus.text_field.vocab))
    print('-' * 60)
    print('Padding Index:', corpus.get_padding_idx())
    print('-' * 60)

    train_iter, valid_iter, test_iter = corpus.get_iterators((256, 256, 256))
    print('Train Iterator:')
    for batch in train_iter:
        print(batch)
        print('-' * 60, '\n')
        break
    print('Valid Iterator:')
    for batch in valid_iter:
        print(batch)
        print('-' * 60, '\n')
        break
    print('Test Iterator:')
    for batch in test_iter:
        print(batch)
        print('-' * 60, '\n')
        break



if __name__ == '__main__':
    test_corpus()
