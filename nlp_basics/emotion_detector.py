# -*- coding: utf-8 -*-
# @Time    : 2020/3/1 下午1:27
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : emotion_detector.py
# @Desc    : 简单的情绪检测

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


class EmotionDetector:
    """情绪检测类
    
    Args:
        train_data_path (Path): 训练文件路径
    
    """
    def __init__(self, train_data_path: os.PathLike):
        raw_data = pd.read_csv(train_data_path, header=None)
        sentences, labels = raw_data[1].values.tolist(), raw_data[0].values.tolist()

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(sentences, labels, test_size=0.2,
                                                                                random_state=0)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.x_train = self.vectorizer.fit_transform(self.x_train)
        self.x_test = self.vectorizer.transform(self.x_test)
        self.model = None

    def train(self):
        """使用逻辑回归训练模型, 通过网格搜索确定超参数"""
        params = {'C': [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 2, 5, 10, 50, 100]}
        self.model = GridSearchCV(LogisticRegression(solver='liblinear'), params, cv=5)
        self.model.fit(self.x_train, self.y_train)
        print('Best Param:', self.model.best_params_)

    def evaluate(self):
        """评估模型"""
        # 测试集分数
        print('Score:', self.model.score(self.x_test, self.y_test))
        # 混淆矩阵
        print('Confusion Matrix:')
        print(confusion_matrix(self.y_test, self.model.predict(self.x_test)))


def main():
    nlp_path = Path('/media/bnu/data/nlp-practice')
    train_data_path = nlp_path / 'emotion-detector/ISEAR.csv'

    detector = EmotionDetector(train_data_path)
    print('Train Data Sample:')
    print(detector.x_train[0])
    print('-' * 100)

    detector.train()
    detector.evaluate()


if __name__ == '__main__':
    main()
