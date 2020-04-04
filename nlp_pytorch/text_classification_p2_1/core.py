# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 下午4:10
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : core.py
# @Desc    : 常用类型和函数

import os
import random
import time
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import tqdm
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator

CPU_COUNT = os.cpu_count()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_COUNT = torch.cuda.device_count()


def set_random_seed(seed: int = 2020):
    """设定程序使用过程中的随机种子
    
    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show_device_info():
    """打印设备信息"""
    print('PyTorch Version:', torch.__version__)
    print('-' * 60)
    if torch.cuda.is_available():
        print('CUDA Device Count:', torch.cuda.device_count())
        print('CUDA Device Name:')
        for i in range(torch.cuda.device_count()):
            print('\t', torch.cuda.get_device_name(i))
        print('CUDA Current Device Index:', torch.cuda.current_device())
        print('-' * 60)


def get_current_time():
    """获取当前时间

    Returns:
        result (str): 当前时间的字符串表示
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
