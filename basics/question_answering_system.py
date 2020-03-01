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
from queue import PriorityQueue
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class QASystem:
    """问答系统类

    Args:
        corpus_path (os.PathLike): 问答系统语料库路径
        glove_path (os.PathLike): GloVe词向量路径

    Attributes:
        question_list (List): 问题列表
        answer_list (List): 答案列表
        word2vec (np.ndarray): 加载后的词向量
        word_count (Dict): 问题中每个单词的出现次数
    """
    def __init__(self, corpus_path: os.PathLike, glove_path: os.PathLike):
        self.corpus_path = corpus_path
        self.golve_path = glove_path

        # 从语料库中获取问答列表
        self.question_list, self.answer_list = self._read_corpus()

        # 加载GloVe词向量
        self.word2vec = self._load_word2vec()

        # 分词统计
        self.word_count = defaultdict(int)
        self.word_tokenizer = WordPunctTokenizer()
        for question in self.question_list:
            word_list = self.word_tokenizer.tokenize(question)
            for word in word_list:
                self.word_count[word] += 1

        # TF-IDF对问题列表进行向量表示
        self.tfidf_vectorizer = TfidfVectorizer(input='content', lowercase=True, stop_words='english', use_idf=True)
        self.x_tfidf = self.tfidf_vectorizer.fit_transform(self.question_list)

        # 构建倒排表
        self.inverted_idx = defaultdict(set)
        for i, question in enumerate(self.question_list):
            word_list = self.word_tokenizer.tokenize(question)
            for word in word_list:
                if self.word_count[word] < 5000:
                    self.inverted_idx[word].add(i)

        # 基于GloVe构建句子向量
        sentence_array = []
        for question in self.question_list:
            word_list = self.word_tokenizer.tokenize(question)
            word_array = []
            for word in word_list:
                if word in self.word2vec:
                    word_array.append(self.word2vec[word])
            word_array = np.vstack(word_array)
            # 对所有词向量求平均获得句子向量
            sentence_array.append(word_array.mean(axis=0))
        self.x_glove = np.vstack(sentence_array)

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

    def _load_word2vec(self) -> Dict[str, np.ndarray]:
        """加载词向量数据

        Returns:
            word2vec (Dict): 词向量字典
        """
        word2vec = {}
        with open(self.golve_path, 'r') as f:
            for line in f:
                temp = line.split()
                word2vec[temp[0]] = np.asarray(temp[1:], dtype='float32')
        return word2vec

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

    def get_top5_answers(self, question: str) -> List[str]:
        """获取指定问题的前5名答案

        Args:
            question (str): 输入问题字符串

        Returns:
            result (List): 问题答案列表
        """
        # 计算余弦相似度
        y = self.tfidf_vectorizer.transform([question])
        similarity_degrees = cosine_similarity(self.x_tfidf, y)
        similarity_degrees = similarity_degrees.squeeze()

        # 通过优先队列进行排序处理, 找出前5的答案
        queue = PriorityQueue()
        for i, value in enumerate(similarity_degrees):
            queue.put_nowait((-value, i))
        result = []
        for _ in range(5):
            _, index = queue.get()
            result.append(self.answer_list[index])
        return result

    def get_top5_answers_invidx(self, question: str) -> List[str]:
        """获取倒排表优化后的前5名答案

        Args:
            question (str): 输入问题字符串

        Returns:
            result (List): 问题答案列表
        """
        # 利用倒排表获取候选集合
        word_list = self.word_tokenizer.tokenize(question)
        index_set = set()
        for word in word_list:
            index_set |= self.inverted_idx[word]
        index_list = list(index_set)

        # 计算余弦相似度
        y = self.tfidf_vectorizer.transform([question])
        similarity_degrees = cosine_similarity(self.x_tfidf[index_list], y)
        similarity_degrees = similarity_degrees.squeeze()

        # 通过优先队列进行排序处理, 找出前5的答案
        queue = PriorityQueue()
        for i, value in zip(index_list, similarity_degrees):
            queue.put_nowait((-value, i))
        result = []
        for _ in range(5):
            _, index = queue.get()
            result.append(self.answer_list[index])
        return result

    def get_top5_answers_emb(self, question: str) -> List[str]:
        """获取使用GloVe后的前5名答案

        Args:
            question (str): 输入问题字符串

        Returns:
            result (List): 问题答案列表
        """
        # 利用倒排表获取候选集合
        word_list = self.word_tokenizer.tokenize(question)
        index_set = set()
        for word in word_list:
            index_set |= self.inverted_idx[word]
        index_list = list(index_set)

        # 生成GloVe对应的问题向量
        word_list = self.word_tokenizer.tokenize(question)
        word_array = []
        for word in word_list:
            if word in self.word2vec:
                word_array.append(self.word2vec[word])
        word_array = np.vstack(word_array)
        y = word_array.mean(axis=0)

        # 计算余弦相似度
        similarity_degrees = cosine_similarity(self.x_glove[index_list], y.reshape(1, -1))
        similarity_degrees = similarity_degrees.squeeze()

        # 通过优先队列进行排序处理, 找出前5的答案
        queue = PriorityQueue()
        for i, value in zip(index_list, similarity_degrees):
            queue.put_nowait((-value, i))
        result = []
        for _ in range(5):
            _, index = queue.get()
            result.append(self.answer_list[index])
        return result


def main():
    nlp_path = Path('/media/bnu/data/nlp-practice')
    corpus_path = nlp_path / 'question-answering-system/train-v2.0.json'
    glove_path = nlp_path / 'word-vector/glove.6B.100d.txt'

    qa = QASystem(corpus_path, glove_path)
    print('Total Words in Question List:', len(qa.word_count))
    print('-' * 100)

    # 语料库中的问答样例
    for q, a in zip(qa.question_list[:10], qa.answer_list[:10]):
        print('Question:', q)
        print('Answer:', a)
    print('-' * 100)

    # 绘图后可以看出是幂律分布
    qa.plot_word_count()
    print('Top10 Words in Question List:', qa.get_top10_words())
    print('-' * 100)

    # TF-IDF后的稀疏度
    sparsity = 1 - qa.x_tfidf.count_nonzero() / (qa.x_tfidf.shape[0] * qa.x_tfidf.shape[1])
    print('X_TF-IDF Sparsity:', sparsity)
    print('-' * 100)

    # 基本查询
    print('Top5 Result:')
    question = 'In what R&B group was she the lead singer?'
    print('Question:', question)
    print('Answers:', qa.get_top5_answers(question))
    print('-' * 100)

    # 倒排表优化后的查询
    print('Top5 Result with Inverted Index:')
    question = 'What album made her a worldwide known artist?'
    print('Question:', question)
    print('Answers:', qa.get_top5_answers_invidx(question))
    print('-' * 100)

    # 使用GloVe后的查询
    print('Top5 Result with GloVe:')
    question = 'In which decade did Beyonce become famous?'
    print('Question:', question)
    print('Answers:', qa.get_top5_answers_emb(question))
    print('-' * 100)


if __name__ == '__main__':
    main()
