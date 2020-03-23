# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 下午8:20
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : learner.py
# @Desc    : 语言模型训练

from collections import defaultdict

import torch
from tqdm import tqdm

from data import BobSueLMDataSet, Corpus
from models import LSTMLM, MaskCrossEntropyLoss
from utils import lm_collate_fn


class LMLearner:
    """语言模型训练学习器
    
    Args:
        corpus (Corpus): 语料库实例
        n_embed (int): 词向量维度
        n_hidden (int): LSTM隐含状态维度
        dropout (float): dropout概率
        rnn_type (str): RNN类型'LSTM'或'GRU'
        batch_size (int): 批处理大小
        early_stopping_round (int): EarlyStopping轮数
        learning_rate (float): 学习率
    """
    def __init__(self, corpus: Corpus, n_embed: int = 200, n_hidden: int = 200, dropout: float = 0.5,
                 rnn_type: str = 'LSTM', batch_size: int = 128, early_stopping_round: int = 5,
                 learning_rate: float = 1e-3):
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.corpus = corpus
        self.batch_size = batch_size
        self.early_stopping_round = early_stopping_round
        self.model = LSTMLM(len(corpus.vocab), n_embed, n_hidden, dropout, rnn_type).to(device)
        self.criterion = MaskCrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = defaultdict(list)

    def fit(self, num_epochs):
        # 定义训练集dataloader
        train_set = BobSueLMDataSet(self.corpus.train_data)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True,
                                                   collate_fn=lm_collate_fn)

        # 定义验证集dataloader
        valid_set = BobSueLMDataSet(self.corpus.valid_data)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=self.batch_size, shuffle=True,
                                                   collate_fn=lm_collate_fn)

        # 记录验证集没有提高的轮数，用于EarlyStopping
        no_improve_round = 0

        for epoch in range(num_epochs):
            train_loss, train_acc, train_words = self._make_train_step(epoch, train_loader)
            # 记录训练信息
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            valid_loss, valid_acc, valid_words = self._make_valid_step(epoch, valid_loader)
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_acc'].append(valid_acc)

            # 根据验证集的准确率进行EarlyStopping
            if self.history['valid_acc'][-1] < max(self.history['valid_acc']):
                no_improve_round += 1
            else:
                no_improve_round = 0
                # 模型存储
                torch.save(self.model.state_dict(), './models/lm-best.pth')

            if no_improve_round == self.early_stopping_round:
                print(f'Early Stopping at Epoch {epoch + 1}')
                break

        return self.history

    def predict(self):
        test_set = BobSueLMDataSet(self.corpus.test_data)
        # 这里注意，为了方便之后分析不要shuffle，batch_size设置为1
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False,
                                                  collate_fn=lm_collate_fn)

        # 读取最佳参数模型
        self.model.load_state_dict(torch.load('./models/lm-best.pth'))
        # 验证模式
        self.model.eval()

        # 总损失
        total_loss = 0.0
        # 正确预测的数目，单词总数
        total_correct, total_words = 0, 0
        # 预测结果字典，包含preds和targets
        test_result = defaultdict(list)

        with torch.no_grad():
            for inputs, targets, lengths, mask in test_loader:
                # 计算模型输出
                outputs = self.model(inputs, lengths)

                # 统计当前预测正确的数目
                total_correct += (outputs.argmax(-1) == targets).sum().item()
                # 统计当前总预测单词数
                total_words += torch.sum(lengths).item()

                # 记录结果
                test_result['preds'].append(outputs.argmax(-1).data.cpu().numpy()[0])
                test_result['targets'].append(targets.data.cpu().numpy()[0])

                # 计算模型Mask交叉熵损失
                loss = self.criterion(outputs, targets, mask)
                # 统计总损失
                total_loss += loss.item() * torch.sum(mask).item()
        return total_loss / total_words, total_correct / total_words, total_words, test_result

    def _make_train_step(self, epoch, train_loader):
        # 训练模式
        self.model.train()

        # 总损失
        total_loss = 0.0
        # 正确预测的数目，单词总数
        total_correct, total_words = 0, 0

        pbar = tqdm(train_loader)

        for inputs, targets, lengths, mask in pbar:
            # 计算模型输出
            outputs = self.model(inputs, lengths)

            # 统计当前预测正确的数目
            total_correct += (outputs.argmax(-1) == targets).sum().item()
            # 统计当前总预测单词数
            total_words += torch.sum(lengths).item()

            # 计算模型Mask交叉熵损失
            loss = self.criterion(outputs, targets, mask)
            # 统计总损失
            total_loss += loss.item() * torch.sum(mask).item()

            pbar.set_description(f'Epoch {epoch + 1} --> Train')
            pbar.set_postfix(loss=total_loss / total_words, acc=total_correct / total_words)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss / total_words, total_correct / total_words, total_words

    def _make_valid_step(self, epoch, valid_loader):
        # 验证模式
        self.model.eval()

        # 总损失
        total_loss = 0.0
        # 正确预测的数目，单词总数
        total_correct, total_words = 0, 0

        with torch.no_grad():
            pbar = tqdm(valid_loader)

            for inputs, targets, lengths, mask in pbar:
                # 计算模型输出
                outputs = self.model(inputs, lengths)

                # 统计当前预测正确的数目
                total_correct += (outputs.argmax(-1) == targets).sum().item()
                # 统计当前总预测单词数
                total_words += torch.sum(lengths).item()

                # 计算模型Mask交叉熵损失
                loss = self.criterion(outputs, targets, mask)
                # 统计总损失
                total_loss += loss.item() * torch.sum(mask).item()

                pbar.set_description(f'Epoch {epoch + 1} --> Valid')
                pbar.set_postfix(loss=total_loss / total_words, acc=total_correct / total_words)

        return total_loss / total_words, total_correct / total_words, total_words
