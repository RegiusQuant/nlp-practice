# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 上午11:46
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : spelling_correction.py
# @Desc    : 简单的拼写纠错

import os
import string
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from nltk.corpus import reuters
from nltk.tokenize import WordPunctTokenizer


class SpellingCorrection:
    """拼写纠错类

    Args:
        vocab_path (Path): 词典库路径
        spell_error_path (Path): 拼写错误语料库路径

    Attributes:
        vocab (Set): 英文单词集合
        vocab_size (int): 单词表大小
        unigram_count (Dict): 一元计数
        bigram_count (Dict): 二元计数
        channel_prob (Dict): 拼写错误概率
    """

    def __init__(self, vocab_path: os.PathLike, spell_error_path: os.PathLike):
        # 构建词典库
        with open(vocab_path) as f:
            self.vocab = {line.strip() for line in f}
        self.vocab_size = len(self.vocab)

        # 获取语料库, 构建语言模型
        categories = reuters.categories()
        corpus = reuters.sents(categories=categories)
        self.unigram_count, self.bigram_count = defaultdict(int), defaultdict(int)
        for doc in corpus:
            doc = ['<s>'] + doc
            for i in range(1, len(doc)):
                self.unigram_count[doc[i]] += 1
                self.bigram_count[(doc[i - 1], doc[i])] += 1

        # 统计拼写错误概率 P(mistake|correct)
        self.channel_prob = defaultdict(dict)
        with open(spell_error_path) as f:
            for line in f:
                temp = line.split(':')
                correct = temp[0].strip()
                mistakes = [m.strip() for m in temp[1].strip().split(',')]
                for m in mistakes:
                    self.channel_prob[correct][m] = 1. / len(mistakes)

    def generate_candidates(self, word: str) -> List[str]:
        """生成编辑距离为1的候选集合

        Args:
            word (str): 错误的输入单词

        Returns:
            result (List): 候选单词列表
        """
        # Insert操作
        s1 = [word[:i] + c + word[i:] for i in range(len(word) + 1) for c in string.ascii_lowercase]
        # Delete操作
        s2 = [word[:i] + word[i + 1:] for i in range(len(word))]
        # Replace操作
        s3 = [word[:i] + c + word[i + 1:] for i in range(len(word)) for c in string.ascii_lowercase]
        result = [word for word in set(s1 + s2 + s3) if word in self.vocab]
        return result

    def predict(self, test_data_path: os.PathLike) -> List[Tuple[str, str]]:
        """对输入文件进行拼写纠错

        Args:
            test_data_path (os.PathLike): 测试文件路径

        Returns:
            result (List): 包含(错误单词, 正确单词)的列表
        """
        result, tokenizer = [], WordPunctTokenizer()
        with open(test_data_path) as f:
            for line in f:
                words = tokenizer.tokenize(line.strip().split('\t')[2])
                for i, word in enumerate(words):
                    # 对不在单词表中的单词进行纠错
                    if word not in self.vocab and len(word) > 1:
                        candidates = self.generate_candidates(word)
                        # 没有候选词的话暂时忽略, 这个时候可以考虑生产编辑距离为2的单词
                        if len(candidates) == 0:
                            continue
                        # 计算分数: score = log(p(correct)) + log(p(mistake|correct))
                        scores = []
                        for c in candidates:
                            score = 0.
                            # Step1: 计算拼写错误概率
                            if c in self.channel_prob and word in self.channel_prob[c]:
                                score += np.log(self.channel_prob[c][word])
                            else:
                                score += np.log(1e-5)
                            # Step2: 计算语言模型概率
                            if i > 0 and (words[i - 1], c) in self.bigram_count:
                                score += np.log((self.bigram_count[(words[i - 1], c)] + 1.0) /
                                                (self.unigram_count[c] + self.vocab_size))
                            else:
                                score += np.log(1.0 / self.vocab_size)
                            scores.append(score)
                        max_index = scores.index(max(scores))
                        result.append((word, candidates[max_index]))
        return result


def main():
    nlp_path = Path('/media/bnu/data/nlp-practice')
    vocab_path = nlp_path / 'spelling-correction/vocab.txt'
    spell_error_path = nlp_path / 'spelling-correction/spell-errors.txt'
    test_data_path = nlp_path / 'spelling-correction/testdata.txt'

    spelling_correction = SpellingCorrection(vocab_path, spell_error_path)
    print('Vocab Sample:')
    print(list(spelling_correction.vocab)[:10])
    print('-' * 100)
    print('Bigram Sample:')
    print(list(spelling_correction.bigram_count.items())[:10])
    print('-' * 100)
    print('Channel Prob Sample:')
    print(spelling_correction.channel_prob['four'])
    print('-' * 100)

    print('Generate Candidates: ')
    print(spelling_correction.generate_candidates('apple'))
    print('-' * 100)

    print('Spelling Correction:')
    print(spelling_correction.predict(test_data_path))


if __name__ == '__main__':
    main()
