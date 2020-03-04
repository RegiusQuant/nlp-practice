# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 下午1:55
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : text_classification.py
# @Desc    : 基于朴素贝叶斯的新闻文本分类

import os
from collections import Counter
from pathlib import Path
from typing import List, Set

import jieba
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class TextClassification:
    """基于朴素贝叶斯的简单文本分类器

    Args:
        data_path (Path): 语料路径
        mode (str): 分类使用的包
    """
    def __init__(self, data_path: Path, mode='sklearn'):
        self.stop_path = data_path / 'stopwords_cn.txt'
        self.folder_path = data_path / 'SogouC' / 'Sample'
        self.mode = mode

    def make_stop_set(self) -> Set[str]:
        """生成停用词表

        Returns:
            word_set (Set): 停用词集合
        """
        with open(self.stop_path) as f:
            word_set = set([word.strip() for word in f.readlines()])
        return word_set

    def process_text(self, test_size: int = 0.2):
        """处理语料库中的文本信息

        Args:
            test_size (float): 测试集占比

        Returns:
            sorted_words (List): 按照词频从大到小排列的单词列表
            train_words_list (List): 训练文本列表
            test_words_list (List): 测试文本列表
            train_class_list (List): 训练类别列表
            test_class_list (List): 测试类别列表
        """
        words_list, class_list = [], []
        for folder in os.listdir(self.folder_path):
            for text_file in os.listdir(self.folder_path / folder):
                file_path = self.folder_path / folder / text_file
                with open(file_path) as f:
                    content = f.read()

                jieba.enable_parallel(4)  # 开启并行分词
                segs = jieba.lcut(content, cut_all=False)  # 精确模式分词
                jieba.disable_parallel()

                words_list.append(segs)
                class_list.append(folder)

        # 划分训练集与测试集
        train_words_list, test_words_list, train_class_list, test_class_list = train_test_split(
            words_list, class_list, test_size=test_size, random_state=0)
        # 统计词频
        word_count = Counter()
        for words in train_words_list:
            word_count.update(words)
        # 将单词按照词频从大到小排序
        sorted_words = sorted(word_count.keys(), key=lambda x: word_count[x], reverse=True)
        return sorted_words, train_words_list, test_words_list, train_class_list, test_class_list

    def get_feature_words(self, sorted_words: List[str], num_delete: int, stop_word_set: Set[str]):
        """ 获取特征词列表

        Args:
            sorted_words (List): 按照词频从大到小排列的单词列表
            num_delete (int): 删除词频出现最多的N个单词
            stop_word_set (List): 停用词列表

        Returns:
            feature_words (List): 特征词列表
        """
        feature_words = []
        for i in range(num_delete, len(sorted_words)):
            if len(feature_words) > 1000:  # 最多选取1000维
                break
            if not sorted_words[i].isdigit() and sorted_words[i] not in stop_word_set and 1 < len(sorted_words[i]) < 5:
                feature_words.append(sorted_words[i])
        return feature_words

    def get_text_features(self, train_words_list: List, test_words_list: List, feature_words: List[str]):
        """获取文本特征

        Args:
            train_words_list (List): 训练文本列表
            test_words_list (List): 测试文本列表
            feature_words (List): 停用词列表

        Returns:
            train_data (List): 训练特征
            test_data (List): 测试特征
        """
        def helper(text):
            text = set(text)
            if self.mode == 'sklearn':
                return [1 if word in text else 0 for word in feature_words]
            elif self.mode == 'nltk':
                return {word: 1 if word in text else 0 for word in feature_words}

        train_data = [helper(text) for text in train_words_list]
        test_data = [helper(text) for text in test_words_list]
        return train_data, test_data

    def fit(self, train_data, train_class_list, test_data, test_class_list):
        """模型训练

        Args:
            train_data (List): 训练特征
            train_class_list (List): 训练标签
            test_data (List): 测试特征
            test_class_list (List): 测试标签

        Returns:
            test_score (float): 测试集上的准确率
        """
        if self.mode == 'sklearn':
            model = MultinomialNB()
            model.fit(train_data, train_class_list)
            test_score = model.score(test_data, test_class_list)
        elif self.mode == 'nltk':
            train_list = list(zip(train_data, test_class_list))
            test_list = list(zip(test_data, test_class_list))
            classifier = nltk.classify.NaiveBayesClassifier.train(train_list)
            test_score = nltk.classify.accuracy(classifier, test_list)
        return test_score

    def run(self):
        """运行模型"""
        stop_word_set = self.make_stop_set()
        sorted_words, train_words_list, test_words_list, train_class_list, test_class_list = self.process_text()

        # 删除词频最高的N个单词, 计算结果
        num_delete_list = list(range(0, 1000, 20))
        score_list = []
        for num_delete in num_delete_list:
            feature_words = self.get_feature_words(sorted_words, num_delete, stop_word_set)
            train_data, test_data = self.get_text_features(train_words_list, test_words_list, feature_words)
            test_score = self.fit(train_data, train_class_list, test_data, test_class_list)
            score_list.append(test_score)
        print(score_list)

        # 结果评价
        plt.plot(num_delete_list, score_list)
        plt.title('Relationship of Number of Delete and Test Score')
        plt.xlabel('number of delete')
        plt.ylabel('score')
        plt.show()


def main():
    data_path = Path('/media/bnu/data/nlp-practice/text-categorization')
    cate = TextClassification(data_path, mode='sklearn')
    cate.run()


if __name__ == '__main__':
    main()
