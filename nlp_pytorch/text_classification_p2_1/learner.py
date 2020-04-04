# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 下午3:30
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : learner.py
# @Desc    : 说明

from core import *
from callbacks import CallbackContainer, HistoryCallback
from metrics import BinaryAccuracy


class TextClassificationLearner:
    def __init__(self, model, optimizer, padding_idx, save_path, early_stopping_round=5):
        self.model = model
        self.model.to(DEVICE)
        self.optimizer = optimizer
        self.padding_idx = padding_idx
        self.save_path = save_path
        self.early_stopping_round = early_stopping_round

        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = BinaryAccuracy()

        # 设置回调
        self.history = HistoryCallback()
        self.callback_container = CallbackContainer([self.history])
        self.callback_container.set_model(self.model)

    def fit(self, train_iter, valid_iter, n_epochs):
        self.callback_container.on_train_begin(logs={'n_epochs': n_epochs})
        no_improve_round = 0

        for epoch in range(n_epochs):
            ############################################################
            # 模型训练
            ############################################################
            self.model.train()
            self.metric.reset()
            total_loss, total_samples = 0.0, 0
            train_loss, train_acc = 0.0, 0.0

            epoch_logs = {}
            self.callback_container.on_epoch_begin(epoch=epoch, logs=epoch_logs)

            pbar = tqdm.tqdm(train_iter)
            pbar.set_description(f'Epoch {epoch+1} --> Train')
            for i, batch in enumerate(pbar):
                self.callback_container.on_batch_begin(batch=i)

                inputs, lengths = batch.text
                targets = batch.label
                mask = (inputs != self.padding_idx)

                outputs = self.model(inputs, mask)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(targets)
                total_samples += len(targets)
                train_loss = total_loss / total_samples
                train_acc = self.metric(outputs, targets)
                pbar.set_postfix(loss=train_loss, acc=train_acc)

                self.callback_container.on_batch_end(batch=i)
            epoch_logs['train_loss'] = train_loss
            epoch_logs['train_acc'] = train_acc

            ############################################################
            # 模型验证
            ############################################################
            self.model.eval()
            self.metric.reset()
            total_loss, total_samples = 0.0, 0
            valid_loss, valid_acc = 0.0, 0.0

            pbar = tqdm.tqdm(valid_iter)
            pbar.set_description(f'Epoch {epoch+1} --> Valid')
            with torch.no_grad():
                for batch in pbar:
                    inputs, lengths = batch.text
                    targets = batch.label
                    mask = (inputs != self.padding_idx)

                    outputs = self.model(inputs, mask)
                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item() * len(targets)
                    total_samples += len(targets)
                    valid_loss = total_loss / total_samples
                    valid_acc = self.metric(outputs, targets)
                    pbar.set_postfix(loss=valid_loss, acc=valid_acc)

            epoch_logs['valid_loss'] = valid_loss
            epoch_logs['valid_acc'] = valid_acc
            self.callback_container.on_epoch_end(epoch=epoch, logs=epoch_logs)

            # 保存最佳模型
            if self.history.history['valid_acc'][-1] == max(self.history.history['valid_acc']):
                torch.save(self.model.state_dict(), self.save_path)
                no_improve_round = 0
            else:   # Early Stopping
                no_improve_round += 1
                if no_improve_round == self.early_stopping_round:
                    print(f'Early Stop at Epoch {epoch+1}')
                    break

        self.callback_container.on_train_end()
        print('Max Accuracy in Validation: ', max(self.history.history['valid_acc']))

    def predict(self, test_iter):
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        self.metric.reset()
        total_loss, total_samples = 0.0, 0
        test_loss, test_acc = 0.0, 0.0

        with torch.no_grad():
            for batch in test_iter:
                inputs, lengths = batch.text
                targets = batch.label
                mask = (inputs != self.padding_idx)

                outputs = self.model(inputs, mask)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * len(targets)
                total_samples += len(targets)
                test_loss = total_loss / total_samples
                test_acc = self.metric(outputs, targets)

        print(f'Test --> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')
