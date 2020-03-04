# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 下午3:55
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : language_detector.py
# @Desc    : 简单的语种检测

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


class LanguageDetector:
    """简单语种检测器"""
    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 2),  # bi-gram模型
            max_features=1000,  # 最多1000个特征
            lowercase=True,  # 转换为小写
            analyzer='char_wb',  # 对字符进行处理
            preprocessor=self._remove_noise  # 去除噪音
        )

    def _remove_noise(self, raw_text: str):
        """取出噪音
        
        Args:
            raw_text (str): 原始文本

        Returns:
            clean_text (str): 经过去噪后的文本
        """
        pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        clean_text = re.sub(pattern, "", raw_text)
        return clean_text

    def features(self, x):
        return self.vectorizer.transform(x)

    def fit(self, x, y):
        self.vectorizer.fit(x, y)
        self.classifier.fit(self.features(x), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, x, y):
        return self.classifier.score(self.features(x), y)


def main():
    data_path = Path('/media/bnu/data/nlp-practice/language-detector/data.csv')
    raw_data = pd.read_csv(data_path, header=None)
    features = [' '.join(x.split()[1:]) for x in raw_data[0].values]
    labels = raw_data[1].values

    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=0)
    detector = LanguageDetector()
    detector.fit(x_train, y_train)
    print(detector.predict('I love English senence.'))
    print(detector.score(x_test, y_test))


if __name__ == '__main__':
    main()
