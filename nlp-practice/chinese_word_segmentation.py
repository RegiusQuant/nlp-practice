# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 下午12:48
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : chinese_word_segmentation.py
# @Desc    : 简单的中文分词工具

from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd


class ChineseWordSegmentation:
    """中文分词类

    Args:
        chinese_dict_path (Path): 中文词库路径
        word_prob (Dict): 自定义的单词概率字典

    Attributes:
        word_set (Set): 中文词集合
        vocab_size (int): 单词表大小
    """

    def __init__(self, chinese_dict_path: Path, word_prob: Dict[str, float]):
        # 从指定路径中读取词库数据, 并将词库中的词保存在集合中
        raw_data = pd.read_excel(chinese_dict_path, header=None)
        self.word_set = set(raw_data[0].values)
        # 用户自定义的单词概率, 对于没有出现在这里但是存在于字典中的, 统一设定概率为1e-5
        self.word_prob = word_prob
        self.vocab_size = len(self.word_set)

    def word_segment_naive(self, input_string: str) -> List[str]:
        """朴素的分词方法

        通过枚举每一种可能的分词结果, 计算对应的句子概率, 选择概率最高的分子结果输出

        Args:
            input_string (str): 需要分词的输入

        Returns:
            best_segment (List): 最好的分词结果列表
        """

        def generate_segment(segment: List[str], current_string: str):
            """通过递归方式生成句子分词

            Args:
                segment (List): 当前的分词结果
                current_string (): 尚未进行分词的句子
            """
            if len(current_string) == 0:
                segments.append(segment.copy())
            for i in range(len(current_string)):
                if current_string[:i + 1] in self.word_set:
                    segment.append(current_string[:i + 1])
                    generate_segment(segment, current_string[i + 1:])
                    segment.pop()

        # Step1: 生成所有分词结果, 并存储在segments中
        segments = []
        generate_segment([], input_string)

        # Step2: 循环所有分词结果, 找出概率最高的分词
        best_segment, best_score = None, float('inf')
        for seg in segments:
            current_score = 0.
            for word in seg:
                current_score += -np.log(self.word_prob.get(word, 1e-5))
            if current_score < best_score:
                best_segment = seg
                best_score = current_score
        return best_segment

    def word_segment_viterbi(self, input_string: str) -> List[str]:
        """基于维特比(Viterbi)算法的分词方法

        通过动态规划来寻找最优分词路径

        Args:
            input_string (str): 需要分词的输入

        Returns:
            best_segment (List): 最好的分词结果列表
        """
        # 初始化动态规划状态(best_pos, best_score)列表
        dp = [[-1, float('inf')] for _ in range(len(input_string))]
        # 循环遍历input_string更新最优概率
        for i in range(len(input_string)):
            # 计算从字符串开始到i作为一个词的概率
            if input_string[:i + 1] in self.word_set:
                dp[i][1] = -np.log(self.word_prob.get(input_string[:i + 1], 1e-5))
            # 计算从j到i作为一个词的概率
            for j in range(1, i + 1):
                if dp[j - 1][1] < float('inf') and input_string[j:i + 1] in self.word_set:
                    current_score = dp[j - 1][1] + -np.log(self.word_prob.get(input_string[j:i + 1], 1e-5))
                    if current_score < dp[i][1]:
                        dp[i][0] = j - 1
                        dp[i][1] = current_score
        # 根据记录的路径生成结果
        pos, best_segment = len(input_string) - 1, []
        while pos >= 0:
            best_segment.append(input_string[dp[pos][0] + 1:pos + 1])
            pos = dp[pos][0]
        best_segment = list(reversed(best_segment))
        return best_segment


def main():
    nlp_path = Path('/media/bnu/data/nlp-practice')
    chinese_dict_path = nlp_path / 'word-segmentation/综合类中文词库.xlsx'
    word_prob = {
        "北京": 0.03, "的": 0.08, "天": 0.005, "气": 0.005, "天气": 0.06, "真": 0.04, "好": 0.05, "真好": 0.04, "啊": 0.01,
        "真好啊": 0.02, "今": 0.01, "今天": 0.07, "课程": 0.06, "内容": 0.06, "有": 0.05, "很": 0.03, "很有": 0.04, "意思": 0.06,
        "有意思": 0.005, "课": 0.01, "程": 0.005, "经常": 0.08, "意见": 0.08, "意": 0.01, "见": 0.005, "有意见": 0.02, "分歧": 0.04,
        "分": 0.02, "歧": 0.005
    }
    print('Total Prob in Word Prob', sum(word_prob.values()))

    segmentation = ChineseWordSegmentation(chinese_dict_path, word_prob)
    print('Total Words in Chinese Dict:', segmentation.vocab_size)
    print('-' * 100)

    print('Segment Naive Test:')
    print(segmentation.word_segment_naive("北京的天气真好啊"))
    print(segmentation.word_segment_naive("今天的课程内容很有意思"))
    print(segmentation.word_segment_naive("经常有意见分歧"))
    print('-' * 100)

    print('Segment Viterbi Test:')
    print(segmentation.word_segment_viterbi("北京的天气真好啊"))
    print(segmentation.word_segment_viterbi("今天的课程内容很有意思"))
    print(segmentation.word_segment_viterbi("经常有意见分歧"))
    print('-' * 100)


if __name__ == '__main__':
    main()
