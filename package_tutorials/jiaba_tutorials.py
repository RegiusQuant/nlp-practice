# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 下午12:25
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : jiaba_tutorials.py
# @Desc    : jieba的常见使用方法

import time
from pathlib import Path

import jieba
import jieba.analyse
import jieba.posseg


def main():
    # 基本分词函数
    segs = jieba.cut('我在学习自然语言处理')  # 精确模式
    print(list(segs))
    segs = jieba.cut('我在学习自然语言处理', cut_all=True)  # 全模式
    print(list(segs))
    segs = jieba.cut_for_search(  # 搜索引擎模式
        '小明硕士毕业于中国科学院计算所，后在哈佛大学深造'
    )
    print(list(segs))
    segs = jieba.lcut('小明硕士毕业于中国科学院计算所，后在哈佛大学深造')  # lcut返回list
    print(segs)
    print(jieba.lcut('如果放到旧字典中将出错。'))
    jieba.suggest_freq(('中', '将'), True)  # 调节词频, 使其能够被分出来
    print(jieba.lcut('如果放到旧字典中将出错。'))
    print('-' * 100)

    # TF-IDF关键词抽取
    root_path = Path('/media/bnu/data/nlp-practice/jieba-tutorials')
    with open(root_path / 'NBA.txt') as f:
        lines = f.read()
        tags = jieba.analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=())
        print(tags)
    with open(root_path / '西游记.txt') as f:
        lines = f.read()
        tags = jieba.analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=())
        print(tags)
    print('-' * 100)

    # TextRank关键词抽取
    with open(root_path / 'NBA.txt') as f:
        lines = f.read()
        tags = jieba.analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n'))
        print(tags)
    with open(root_path / '西游记.txt') as f:
        lines = f.read()
        tags = jieba.analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        print(tags)
    print('-' * 100)

    # 词性标注
    pseg = jieba.posseg.cut('我爱自然语言处理')
    for word, pos in pseg:
        print(word, pos)
    print('-' * 100)

    # 并行分词
    jieba.enable_parallel(4)
    with open(root_path / '西游记.txt') as f:
        lines = f.read()
        t1 = time.time()
        seg = list(jieba.cut(lines))
        t2 = time.time()
        print('Parallel Speed {} bytes/sec'.format(len(lines) / (t2 - t1)))

    jieba.disable_parallel()
    with open(root_path / '西游记.txt') as f:
        lines = f.read()
        t1 = time.time()
        segs = list(jieba.cut(lines))
        t2 = time.time()
        print('Non-Parallel Speed {} bytes/sec'.format(len(lines) / (t2 - t1)))
    print('-' * 100)

    # 词语在原文的起止位置
    tokens = jieba.tokenize('自然语言处理非常有用')   # 默认模式
    for token in tokens:
        print('{}\t\t start: {} \t\t end: {}'.format(token[0], token[1], token[2]))
    tokens = jieba.tokenize('自然语言处理非常有用', mode='search')   # 搜索模式
    print('-' * 100)
    for token in tokens:
        print('{}\t\t start: {} \t\t end: {}'.format(token[0], token[1], token[2]))


if __name__ == '__main__':
    main()
