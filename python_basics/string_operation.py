# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 上午10:47
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : string_operation.py
# @Desc    : 字符串操作

import string


def main():
    # 去除空格及特殊符号
    test_string = 'Hello, World!'
    print(test_string.strip())
    print(test_string.lstrip('Hello, '))
    print(test_string.rstrip('!'))
    print('-' * 100)

    # 连接字符串
    test_string_1 = 'test1'
    test_string_2 = 'test2'
    test_string_1 += test_string_2
    print(test_string_1)
    print('-' * 100)

    # 查找字符
    test_string = 'Hello'
    print(test_string.index('l'))
    try:
        print(test_string.index('b'))
    except ValueError:
        print('Not Found')
    print('-' * 100)

    # 比较字符串
    test_string_1 = 'test'
    test_string_2 = 'test2'
    print(test_string_1 < test_string_2)
    print('-' * 100)

    # 大小写转换
    test_string = 'Hello, World!'
    print(test_string.lower())
    print('-' * 100)

    # 翻转字符串
    test_string = 'abcdefg'
    print(test_string[::-1])
    print('-' * 100)

    # 查找字符串
    test_string = 'Hello, World'
    print(test_string.find('llo'))
    print('-' * 100)

    # 分割字符串
    test_string = 'ab,cde,fg,hij'
    print(test_string.split(','))
    print('-' * 100)

    # 统计出现次数最多的字母
    test_string = 'Hello, World'
    test_string.lower()
    print(max(string.ascii_lowercase, key=test_string.count))
    print('-' * 100)


if __name__ == '__main__':
    main()
