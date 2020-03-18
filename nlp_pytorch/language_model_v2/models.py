# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 下午12:46
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : models.py
# @Desc    : 语言模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegSampleLM(nn.Module):
    """负例采样语言模型结构

    Args:
        n_words (int): 词表大小
        n_embed (int): 嵌入层大小
        dropout (float): Dropout概率
    """
    def __init__(self, n_words, n_embed=200, dropout=0.5):
        super(NegSampleLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        # 输入的Embedding
        self.embed_in = nn.Embedding(n_words, n_embed)
        # 输出的Embedding
        self.embed_out = nn.Embedding(n_words, n_embed)
        # 这里embed_size一定要和hidden_size相同，为了之后点积计算loss
        self.lstm = nn.LSTM(n_embed, n_embed, batch_first=True)

    def forward(self, inputs, poss, negs, lengths, mask):
        # x_embed: (batch_size, seq_len, embed_size)
        x_embed = self.drop(self.embed_in(inputs))
        # poss_embed: (batch_size, seq_len, embed_size)
        poss_embed = self.embed_out(poss)
        # negs_embed: (batch_size, seq_len, n_negs, embed_size)
        negs_embed = self.embed_out(negs)

        x_embed = nn.utils.rnn.pack_padded_sequence(x_embed, lengths, batch_first=True, enforce_sorted=False)
        # x_lstm: (batch_size, seq_len, embed_size)
        x_lstm, _ = self.lstm(x_embed)
        x_lstm, _ = nn.utils.rnn.pad_packed_sequence(x_lstm, batch_first=True)

        # x_lstm: (batch_size * seq_len, embed_size, 1)
        x_lstm = x_lstm.view(-1, x_lstm.shape[2], 1)

        # poss_embed: (batch_size * seq_len, 1, embed_size)
        poss_embed = poss_embed.view(-1, 1, poss_embed.shape[2])
        # negs_embed: (batch_size * seq_len, n_negs, embeds)
        negs_embed = negs_embed.view(-1, negs_embed.shape[2], negs_embed.shape[3])

        # poss_mm: (batch_size * seq_len)
        poss_mm = torch.bmm(poss_embed, x_lstm).squeeze()
        # negs_mm: (batch_size * seq_len, n_negs)
        negs_mm = torch.bmm(negs_embed, -x_lstm).squeeze()

        mask = mask.view(-1)
        poss_loss = F.logsigmoid(poss_mm) * mask
        negs_loss = F.logsigmoid(negs_mm).mean(1) * mask

        total_loss = -(poss_loss + negs_loss)
        return total_loss.mean(), x_lstm


class ContextLM(nn.Module):
    """基于上下文的语言模型结构

    Args:
        n_words (int): 词表大小
        n_embed (int): 嵌入层大小
        n_hidden (int): 隐含层大小
        dropout (float): Dropout概率
    """
    def __init__(self, n_words, n_embed=200, n_hidden=200, dropout=0.5):
        super(ContextLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(n_words, n_embed)
        self.encoder = nn.LSTM(n_embed, n_hidden, batch_first=True)
        self.decoder = nn.LSTM(n_embed, n_hidden, batch_first=True)
        self.linear = nn.Linear(n_hidden, n_words)

    def forward(self, contexts, inputs, ctx_lengths, inp_lengths):
        # 对上一句话进行编码
        ctx_emb = self.drop(self.embed(contexts))
        ctx_emb = nn.utils.rnn.pack_padded_sequence(ctx_emb, ctx_lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.encoder(ctx_emb)

        # 对当前句子进行预测
        inp_emb = self.drop(self.embed(inputs))
        inp_emb = nn.utils.rnn.pack_padded_sequence(inp_emb, inp_lengths, batch_first=True, enforce_sorted=False)
        inp_out, _ = self.decoder(inp_emb, (h_n, c_n))
        inp_out, _ = nn.utils.rnn.pad_packed_sequence(inp_out, batch_first=True)

        return self.linear(self.drop(inp_out))


class MaskCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskCrossEntropyLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets, mask):
        # outputs shape: (batch_size * max_len, vocab_size)
        outputs = outputs.view(-1, outputs.size(2))
        # targets shape: (batch_size * max_len)
        targets = targets.view(-1)
        # mask shape: (batch_size * max_len)
        mask = mask.view(-1)
        loss = self.celoss(outputs, targets) * mask
        return torch.sum(loss) / torch.sum(mask)