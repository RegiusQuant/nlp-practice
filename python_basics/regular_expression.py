# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 下午12:38
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : regular_expression.py
# @Desc    : 正则表达式

import re


def main():
    # re使用的基本步骤
    pattern = re.compile(r'Hello.*!')
    match = pattern.match('Hello, Regius! How are You?')
    if match:
        print(match.group())
    print('-' * 100)

    # match的常见属性和方法
    m = re.match(r'(\w+) (\w+)(?P<sign>.*)', 'Hello Regius!')

    print('m.string:', m.string)
    print('m.re:', m.re)
    print('m.pos:', m.pos)
    print('m.endpos:', m.endpos)
    print('m.lastindex:', m.lastindex)
    print('m.lastgroup:', m.lastgroup)

    print('m.group(1, 2):', m.group(1, 2))
    print('m.groups():', m.groups())
    print('m.groupdict():', m.groupdict())
    print('m.start(2):', m.start(2))
    print('m.end(2):', m.end(2))
    print('m.span(2):', m.span(2))
    print(r"m.expand(r'\2 \1\3'):", m.expand(r'\2 \1\3'))
    print('-' * 100)

    # pattern的常见属性
    p = re.compile(r'(\w+) (\w+)(?P<sign>.*)')
    print('p.pattern:', p.pattern)
    print('p.flags:', p.flags)
    print('p.groups', p.groups)
    print('p.groupindex:', p.groupindex)
    print('-' * 100)

    # pattern的使用
    pattern = re.compile(r'R.*s')
    match = pattern.search('Hello Regius!')  # search用于查找字串, 这个例子中match方法无法成功匹配
    if match:
        print(match.group())

    pattern = re.compile(r'\d+')
    print(pattern.split('one1two2three3four4'))
    print(pattern.findall('one1two2three3four4'))
    for m in pattern.finditer('one1two2three3four4'):
        print(m.group())

    pattern = re.compile(r'(\w+) (\w+)')
    test_string = 'come on, hello regius!'
    print(pattern.sub(r'\2 \1', test_string))
    print(pattern.subn(r'\2 \1', test_string))

    def func(m: re.Match):
        return m.group(1).title() + ' ' + m.group(2).title()

    print(pattern.sub(func, test_string))
    print(pattern.subn(func, test_string))


if __name__ == '__main__':
    main()
