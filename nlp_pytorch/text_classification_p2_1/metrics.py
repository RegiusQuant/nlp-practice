# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 下午5:20
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : metrics.py
# @Desc    : 评估指标

from core import *


class BaseMetric(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        raise NotImplementedError('custom metric must implement this function.')

    @abstractmethod
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError('custom metric must implement this function.')


class BinaryAccuracy(BaseMetric):
    """二分类准确率"""
    def __init__(self):
        super(BinaryAccuracy, self).__init__()
        self.correct_count = 0
        self.total_count = 0

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        preds = torch.round(torch.sigmoid(outputs))
        self.correct_count += (preds == targets).int().sum().item()
        self.total_count += len(targets)
        acc = self.correct_count / self.total_count
        return np.round(acc, 4)
