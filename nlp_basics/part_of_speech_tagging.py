# -*- coding: utf-8 -*-
# @Time    : 2020/2/24 上午10:52
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : part_of_speech_tagging.py
# @Desc    : 简单的词性标注

import os
from pathlib import Path
from typing import List

import numpy as np


class POSTagger:
    """词性标注类

    Args:
        train_data_path (Path): 训练文件路径

    Attributes:
        word_to_idx (Dict): 单词->编号的字典
        tag_to_idx (Dict): 词性->编号的字典
        idx_to_word (Dict): 编号->单词的字典
        idx_to_tag (Dict): 编号->词性的字典
        num_words (int): 单词表大小
        num_tags (int): 词性表大小
    """

    def __init__(self, train_data_path: os.PathLike):
        # 初始化字典和统计信息
        self.word_to_idx, self.tag_to_idx = {}, {}
        with open(train_data_path, 'r') as f:
            for line in f:
                items = line.split('/')
                word, tag = items[0].strip(), items[1].strip()
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
                if tag not in self.tag_to_idx:
                    self.tag_to_idx[tag] = len(self.tag_to_idx)
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}
        self.num_words = len(self.idx_to_word)
        self.num_tags = len(self.idx_to_tag)

        # 定义词性在句首的概率P(pos)
        self.pos_start_prob = np.zeros(self.num_tags)
        # 定义给定词性出现单词的概率P(word|pos)
        self.word_pos_prob = np.zeros((self.num_tags, self.num_words))
        # 定义词性转移概率P(curr_pos|prev_pos)
        self.pos_trans_prob = np.zeros((self.num_tags, self.num_tags))

        # 统计计数
        with open(train_data_path, 'r') as f:
            # 记录前一个tag, 用于判断是否为句首
            prev_tag = '.'
            for line in f:
                items = line.split('/')
                word_idx, tag_idx = (self.word_to_idx[items[0].strip()], self.tag_to_idx[items[1].strip()])
                if prev_tag == '.':  # 句子开头
                    self.pos_start_prob[tag_idx] += 1
                prev_tag_idx = self.tag_to_idx[prev_tag]
                self.pos_trans_prob[prev_tag_idx][tag_idx] += 1
                self.word_pos_prob[tag_idx][word_idx] += 1
                prev_tag = items[1].strip()
        # 将计数结果转换为概率
        self.pos_start_prob = self.pos_start_prob / self.pos_start_prob.sum()
        self.word_pos_prob = self.word_pos_prob / self.word_pos_prob.sum(axis=1).reshape(-1, 1)
        self.pos_trans_prob = self.pos_trans_prob / self.pos_trans_prob.sum(axis=1).reshape(-1, 1)

    def viterbi(self, s: str) -> List[str]:
        """ 使用维特比算法获得词性标注

        Args:
            s (str): 输入字符串, 使用单词见空格进行分割

        Returns:
            tag_list (List): 词性标注列表
        """
        word_idx_list = [self.word_to_idx[c] for c in s.split(' ')]
        length = len(word_idx_list)

        # 动态规划状态转移矩阵
        dp = np.zeros((length, self.num_tags))
        # 动态规划路径记录矩阵
        prev = -np.ones((length, self.num_tags), dtype=np.int)
        # 定义简单平滑后的log函数
        log = lambda x: np.log(x) if x != 0 else np.log(1e-6)

        # 初始化状态
        for j in range(self.num_tags):
            dp[0][j] = log(self.pos_start_prob[j]) + log(self.word_pos_prob[j][word_idx_list[0]])
        # 状态转移
        for i in range(1, length):
            for j in range(self.num_tags):
                dp[i][j] = float('-inf')
                for k in range(self.num_tags):
                    score = (dp[i - 1][k] + log(self.pos_trans_prob[k][j]) +
                             log(self.word_pos_prob[j][word_idx_list[i]]))
                    if score > dp[i][j]:
                        dp[i][j] = score
                        prev[i][j] = k

        # 获得路径
        best_idx_list = [0] * length
        best_idx_list[-1] = np.argmax(dp[-1])
        for i in range(length - 2, -1, -1):
            best_idx_list[i] = prev[i + 1][best_idx_list[i + 1]]
        tag_list = [self.idx_to_tag[i] for i in best_idx_list]
        return tag_list


def main():
    nlp_path = Path('/media/bnu/data/nlp-practice')
    train_data_path = nlp_path / 'part-of-speech-tagging/traindata.txt'

    pos_tagger = POSTagger(train_data_path)
    print('Word to Index:')
    print(pos_tagger.word_to_idx)
    print('Tag to Index:')
    print(pos_tagger.tag_to_idx)
    print('Index to Word')
    print(pos_tagger.idx_to_word)
    print('Index to Tag:')
    print(pos_tagger.idx_to_tag)
    print('Number of Words:', pos_tagger.num_words)
    print('Number of Tags:', pos_tagger.num_tags)
    print('-' * 100)

    print('Pos Transfer Prob:')
    print(pos_tagger.pos_trans_prob)
    print('-' * 100)

    s = "Social Security number , passport number and details about the services provided for the payment"
    print(pos_tagger.viterbi(s))


if __name__ == '__main__':
    main()
