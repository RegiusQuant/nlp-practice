# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 下午5:10
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : callbacks.py
# @Desc    : 模型训练的回调

from core import *


class BaseCallback:
    """回调基类,用于各个回调类的继承"""

    def __init__(self):
        self.model = None
        self.params = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs: Optional[Dict] = None):
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        pass


class CallbackContainer:
    """保存一系列回调的容器"""

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        self.callbacks = []
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type):
                    self.callbacks.append(callback())
                else:
                    self.callbacks.append(callback)
        self.model = None

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self, logs: Optional[Dict] = None):
        logs = logs or {}
        logs['start_time'] = get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        logs = logs or {}
        logs['end_time'] = get_current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class HistoryCallback(BaseCallback):
    """记录训练历史的回调"""

    def __init__(self):
        super(HistoryCallback, self).__init__()
        self.epochs: List[int] = []
        self.history: Dict[str, List[float]] = defaultdict(list)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        logs = deepcopy(logs) or {}
        for k, v in logs.items():
            self.history[k].append(v)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        self.epochs.append(epoch)
        logs = deepcopy(logs) or {}
        for k, v in logs.items():
            self.history[k].append(v)
