# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 下午3:24
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : models.py
# @Desc    : 情感分析模型

from core import *
from data import SSTCorpus


class EmbedAvgModel(nn.Module):
    """词向量平均模型
    
    Args:
        n_words (int): 词表大小
        n_embed (int): Embedding维度
        p_drop (float): Dropout概率
        padding_idx (int): <pad>的索引
    """
    def __init__(self, n_words: int, n_embed: int, p_drop: float, padding_idx: int):
        super(EmbedAvgModel, self).__init__()
        self.embed = nn.Embedding(n_words, n_embed, padding_idx=padding_idx)
        self.linear = nn.Linear(n_embed, 1)
        self.drop = nn.Dropout(p_drop)

    def forward(self, inputs, mask):
        # (batch, len, n_embed)
        inp_embed = self.drop(self.embed(inputs))
        # (batch, len, 1)
        mask = mask.float().unsqueeze(2)
        # (batch, len, n_embed)
        inp_embed = inp_embed * mask
        # (batch, n_embed)
        sum_embed = inp_embed.sum(1) / (mask.sum(1) + 1e-5)
        return self.linear(sum_embed).squeeze()


class AttnAvgModel(nn.Module):
    """Attention加权平均模型

    Args:
        n_words (int): 词表大小
        n_embed (int): Embedding维度
        p_drop (float): Dropout概率
        padding_idx (int): <pad>的索引
    """
    def __init__(self, n_words, n_embed, p_drop, padding_idx):
        super(AttnAvgModel, self).__init__()
        self.embed = nn.Embedding(n_words, n_embed, padding_idx=padding_idx)
        self.linear = nn.Linear(n_embed, 1)
        self.drop = nn.Dropout(p_drop)
        self.coef = nn.Parameter(torch.randn(1, 1, n_embed), requires_grad=True)

    def forward(self, inputs, mask):
        # (batch, len, n_embed)
        inp_embed = self.embed(inputs)
        # (batch, len)
        inp_cos = F.cosine_similarity(inp_embed, self.coef, dim=-1)
        inp_cos.masked_fill_(~mask, -1e5)
        # (batch, 1, len)
        inp_attn = F.softmax(inp_cos, dim=-1).unsqueeze(1)
        # (batch, n_embed)
        sum_embed = torch.bmm(inp_attn, inp_embed).squeeze()
        sum_embed = self.drop(sum_embed)
        return self.linear(sum_embed).squeeze()

    def calc_attention_weight(self, text):
        # (1, len, n_embed)
        inp_embed = self.embed(text)
        # (1, len)
        inp_cos = F.cosine_similarity(inp_embed, self.coef, dim=-1)
        # (batch, 1, len)
        inp_attn = F.softmax(inp_cos, dim=-1)
        return inp_attn


############################################################
# 测试模型
############################################################
def test_model():
    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    corpus = SSTCorpus(data_path)
    train_iter, valid_iter, test_iter = corpus.get_iterators((256, 256, 256))
    model = AttnAvgModel(
        n_words=len(corpus.text_field.vocab),
        n_embed=100,
        p_drop=0.2,
        padding_idx=corpus.get_padding_idx(),
    ).to(DEVICE)

    for batch in train_iter:
        inputs, lengths = batch.text
        mask = (inputs != corpus.get_padding_idx())
        outputs = model(inputs, mask)
        print(outputs.shape)
        break


if __name__ == '__main__':
    test_model()
