# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 下午9:52
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : learner.py
# @Desc    : 情感分析学习器

from collections import defaultdict

import torch
import torch.nn as nn
import tqdm


class SALearner:

    def __init__(self, corpus, model, device):
        self.corpus = corpus
        self.device = device
        self.model = model
        self.model.to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.history = defaultdict(list)

    def calc_correct_count(self, outputs, targets):
        preds = torch.round(torch.sigmoid(outputs))
        return (preds == targets).long().sum().item()

    def fit(self, train_iter, valid_iter, n_epochs):
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            total_samples, total_correct = 0, 0

            pbar = tqdm.tqdm(train_iter)
            pbar.set_description(f'Epoch {epoch + 1} --> Train')

            for batch in pbar:
                inputs, targets = batch.text.to(self.device), batch.label.to(self.device)
                mask = (inputs != self.corpus.text_field.vocab.stoi['<pad>']).float()
                # print(mask)
                outputs = self.model(inputs, mask)
                # print(outputs)

                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_samples += len(targets)
                total_loss += loss.item() * len(targets)
                total_correct += self.calc_correct_count(outputs, targets)

                pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

            self.history['train_loss'].append(total_loss / total_samples)
            self.history['train_acc'].append(total_correct / total_samples)

            self.model.eval()
            total_loss = 0.0
            total_samples, total_correct = 0, 0

            pbar = tqdm.tqdm(valid_iter)
            pbar.set_description(f'Epoch {epoch + 1} --> Valid')

            with torch.no_grad():
                for batch in pbar:
                    inputs, targets = batch.text.to(self.device), batch.label.to(self.device)
                    mask = (inputs != self.corpus.text_field.vocab.stoi['<pad>']).float()
                    outputs = self.model(inputs, mask)

                    loss = self.criterion(outputs, targets)
                    total_samples += len(targets)
                    total_loss += loss.item() * len(targets)
                    total_correct += self.calc_correct_count(outputs, targets)

                    pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

            self.history['valid_loss'].append(total_loss / total_samples)
            self.history['valid_acc'].append(total_correct / total_samples)