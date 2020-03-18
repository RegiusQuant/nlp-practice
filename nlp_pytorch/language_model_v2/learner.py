# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 下午1:19
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : learner.py
# @Desc    : 语言模型学习器

from collections import defaultdict

import torch
import tqdm

from data import NegSampleDataSet, ContextDataset, neglm_collate_fn, ctxlm_collate_fn
from models import NegSampleLM, ContextLM, MaskCrossEntropyLoss


class NegSampleLearner:
    def __init__(self, corpus, n_embed=200, dropout=0.5, n_negs=20, batch_size=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.corpus = corpus
        self.model = NegSampleLM(len(corpus.vocab), n_embed, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.n_negs = n_negs
        self.batch_size = batch_size
        self.history = defaultdict(list)

    def fit(self, num_epochs):
        train_set = NegSampleDataSet(
            self.corpus.train_data,
            self.corpus.word_freqs,
            n_negs=self.n_negs,
        )
        valid_set = NegSampleDataSet(self.corpus.valid_data, self.corpus.word_freqs, n_negs=self.n_negs)

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True,
                                                   collate_fn=neglm_collate_fn)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=self.batch_size, shuffle=False,
                                                   collate_fn=neglm_collate_fn)

        for epoch in range(num_epochs):
            train_loss, train_acc, train_words = self._make_train_step(epoch, train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            valid_loss, valid_acc, valid_words = self._make_valid_step(epoch, valid_loader)
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_acc'].append(valid_acc)

            if self.history['valid_acc'][-1] == max(self.history['valid_acc']):
                torch.save(self.model.state_dict(), './models/neglm-best.pth')

    def _calc_correct(self, model, x_lstm, poss, mask):
        with torch.no_grad():
            x_lstm = x_lstm.squeeze()  # (seq_len, embedding_size)
            embed_weight = model.embed_out.weight.transpose(0, 1)  # (embedding_size, n_words)
            preds = x_lstm @ embed_weight
            preds = preds.argmax(dim=-1)
            preds = preds.view(-1, poss.shape[1])
            return ((preds == poss) * mask).sum().item()

    def _make_train_step(self, epoch, train_loader):
        # 训练模式
        self.model.train()

        # 总损失
        total_loss = 0.0
        # 预测单词总数
        total_words, total_correct = 0, 0

        pbar = tqdm.tqdm(train_loader)
        pbar.set_description(f'Epoch {epoch + 1} --> Train')
        for inputs, poss, negs, lengths, mask in pbar:
            inputs = inputs.to(self.device)
            poss = poss.to(self.device)
            negs = negs.to(self.device)
            lengths = lengths.to(self.device)
            mask = mask.to(self.device)

            # 模型损失
            loss, x_lstm = self.model(inputs, poss, negs, lengths, mask)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计信息
            sent_words = lengths.sum().item()
            total_words += sent_words
            total_correct += self._calc_correct(self.model, x_lstm, poss, mask)
            total_loss += loss.item() * sent_words

            pbar.set_postfix(loss=total_loss / total_words, acc=total_correct / total_words)

        return total_loss / total_words, total_correct / total_words, total_words

    def _make_valid_step(self, epoch, valid_loader):
        # 验证模式
        self.model.eval()

        # 总损失
        total_loss = 0.0
        # 预测正确个数，预测单词总数
        total_correct, total_words = 0, 0

        with torch.no_grad():
            pbar = tqdm.tqdm(valid_loader)
            pbar.set_description(f'Epoch {epoch + 1} --> Valid')

            for inputs, poss, negs, lengths, mask in pbar:
                inputs = inputs.to(self.device)
                poss = poss.to(self.device)
                negs = negs.to(self.device)
                lengths = lengths.to(self.device)
                mask = mask.to(self.device)

                # 模型损失
                loss, x_lstm = self.model(inputs, poss, negs, lengths, mask)

                # 统计信息
                sent_words = lengths.sum().item()
                total_words += sent_words
                total_correct += self._calc_correct(self.model, x_lstm, poss, mask)
                total_loss += loss.item() * sent_words

                pbar.set_postfix(loss=total_loss / total_words, acc=total_correct / total_words)

        return total_loss / total_words, total_correct / total_words, total_words

    def predict(self):
        test_set = NegSampleDataSet(self.corpus.test_data, self.corpus.word_freqs, n_negs=self.n_negs)

        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False,
                                                  collate_fn=neglm_collate_fn)
        # 读取最佳参数模型
        self.model.load_state_dict(torch.load('./models/neglm-best.pth'))
        # 验证模式
        self.model.eval()

        # 预测正确个数，预测单词总数
        total_correct, total_words = 0, 0

        with torch.no_grad():
            for inputs, poss, negs, lengths, mask in test_loader:
                inputs = inputs.to(self.device)
                poss = poss.to(self.device)
                negs = negs.to(self.device)
                lengths = lengths.to(self.device)
                mask = mask.to(self.device)

                # 模型损失
                loss, x_lstm = self.model(inputs, poss, negs, lengths, mask)

                # 统计信息
                sent_words = lengths.sum().item()
                total_correct += self._calc_correct(self.model, x_lstm, poss, mask)
                total_words += sent_words
        return total_correct / total_words, total_words


class ContextLearner:

    def __init__(self, corpus, n_embed=200, n_hidden=200, dropout=0.5,
                 batch_size=128, early_stopping_round=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.corpus = corpus
        self.model = ContextLM(len(corpus.vocab), n_embed, n_hidden, dropout)
        self.model.to(self.device)
        self.criterion = MaskCrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.history = defaultdict(list)
        self.early_stopping_round = early_stopping_round
        self.batch_size = batch_size

    def fit(self, num_epoch):
        train_set = ContextDataset(
            self.corpus.train_data
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctxlm_collate_fn
        )

        valid_set = ContextDataset(
            self.corpus.valid_data
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ctxlm_collate_fn
        )

        no_improve_round = 0

        for epoch in range(num_epoch):
            train_loss, train_acc, train_words = self._make_train_step(epoch, train_loader)
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
                torch.save(self.model.state_dict(), './models/ctxlm-best.pth')
            if no_improve_round == self.early_stopping_round:
                print(f'Early Stopping at Epoch {epoch + 1}')
                break

    def _make_train_step(self, epoch, train_loader):
        self.model.train()

        total_loss = 0.0
        total_correct, total_words = 0, 0

        pbar = tqdm.tqdm(train_loader)
        pbar.set_description(f'Epoch {epoch + 1} --> Train')

        for batch in pbar:
            contexts = batch[0].to(self.device)
            inputs = batch[1].to(self.device)
            targets = batch[2].to(self.device)
            ctx_lengths = batch[3].to(self.device)
            inp_lengths = batch[4].to(self.device)
            mask = batch[5].to(self.device)

            outputs = self.model(contexts, inputs, ctx_lengths, inp_lengths)
            loss = self.criterion(outputs, targets, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_correct += (outputs.argmax(-1) == targets).sum().item()
            total_words += torch.sum(inp_lengths).item()
            total_loss += loss.item() * torch.sum(mask).item()

            pbar.set_postfix(loss=total_loss / total_words, acc=total_correct / total_words)

        return total_loss / total_words, total_correct / total_words, total_words

    def _make_valid_step(self, epoch, valid_loader):
        self.model.eval()

        total_loss = 0.0
        total_correct, total_words = 0, 0

        with torch.no_grad():
            pbar = tqdm.tqdm(valid_loader)
            pbar.set_description(f'Epoch {epoch + 1} --> Valid')

            for batch in pbar:
                contexts = batch[0].to(self.device)
                inputs = batch[1].to(self.device)
                targets = batch[2].to(self.device)
                ctx_lengths = batch[3].to(self.device)
                inp_lengths = batch[4].to(self.device)
                mask = batch[5].to(self.device)

                outputs = self.model(contexts, inputs, ctx_lengths, inp_lengths)
                loss = self.criterion(outputs, targets, mask)

                total_correct += (outputs.argmax(-1) == targets).sum().item()
                total_words += torch.sum(inp_lengths).item()
                total_loss += loss.item() * torch.sum(mask).item()

                pbar.set_postfix(loss=total_loss / total_words, acc=total_correct / total_words)

        return total_loss / total_words, total_correct / total_words, total_words

    def predict(self):
        test_set = ContextDataset(
            self.corpus.test_data
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            collate_fn=ctxlm_collate_fn
        )
        # 读取最佳参数模型
        self.model.load_state_dict(torch.load('./models/ctxlm-best.pth'))
        self.model.eval()

        total_loss = 0.0
        total_correct, total_words = 0, 0
        test_result = defaultdict(list)

        with torch.no_grad():
            for batch in test_loader:
                contexts = batch[0].to(self.device)
                inputs = batch[1].to(self.device)
                targets = batch[2].to(self.device)
                ctx_lengths = batch[3].to(self.device)
                inp_lengths = batch[4].to(self.device)
                mask = batch[5].to(self.device)

                outputs = self.model(contexts, inputs, ctx_lengths, inp_lengths)
                loss = self.criterion(outputs, targets, mask)

                total_correct += (outputs.argmax(-1) == targets).sum().item()
                total_words += torch.sum(inp_lengths).item()
                total_loss += loss.item() * torch.sum(mask).item()

                test_result['preds'].append(outputs.argmax(-1).data.cpu().numpy()[0])
                test_result['targets'].append(targets.data.cpu().numpy()[0])

        return total_loss / total_words, total_correct / total_words, total_words, test_result