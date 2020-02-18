# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 下午3:10
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : question_answering_system.py
# @Desc    : 简单的问答系统

import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer


class QASystem:
    def __init__(self, corpus_path: os.PathLike):
        # 从语料库中获取问答列表
        self.corpus_path = corpus_path
        self.question_list, self.answer_list = self._read_corpus()

        # 分词统计
        self.word_count = defaultdict(int)
        tokenizer = WordPunctTokenizer()
        for question in self.question_list:
            word_list = tokenizer.tokenize(question)
            for word in word_list:
                self.word_count[word] += 1

    def _read_corpus(self) -> Tuple[List[str], List[str]]:
        """从Json语料库中读取问答数据

        Returns:
            question_list, answer_list (List, List): 问题列表和答案列表
        """
        with open(self.corpus_path, 'r') as f:
            json_dict = json.load(f)
        question_list, answer_list = [], []
        for data in json_dict['data']:
            for paragraphs in data['paragraphs']:
                for qas in paragraphs['qas']:
                    if len(qas['answers']) > 0:
                        question_list.append(qas['question'])
                        answer_list.append(qas['answers'][0]['text'])
        return question_list, answer_list

    def plot_word_count(self):
        """将单词的频率从大到小排序并进行绘制"""
        freq_list = sorted(list(self.word_count.values()), reverse=True)
        plt.plot(freq_list[:100])
        plt.ylabel('Frequency')
        plt.show()

    def get_top10_words(self) -> List[str]:
        """获取出现次数前10的单词

        Returns:
            result (List): 出现次数前10的单词列表
        """
        temp_list = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        result = [x[0] for x in temp_list[:10]]
        return result


def main():
    nlp_path = Path('/media/bnu/data/nlp-practice')
    corpus_path = nlp_path / 'question-answering-system/train-v2.0.json'
    qa = QASystem(corpus_path)
    print('Total Words in Question List:', len(qa.word_count))
    print('-' * 100)
    # 绘图后可以看出是幂律分布
    qa.plot_word_count()
    print('Top10 Words in Question List:', qa.get_top10_words())
    print('-' * 100)


if __name__ == '__main__':
    main()
