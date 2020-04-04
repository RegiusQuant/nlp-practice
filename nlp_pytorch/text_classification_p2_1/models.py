# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 下午3:24
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : models.py
# @Desc    : 情感分析模型

from core import *
from data import SSTCorpus


class SimpleSelfAttentionModel(nn.Module):

    def __init__(self, n_words, n_embed, p_drop, pad_idx, res_conn=False, score_fn='dot'):
        super(SimpleSelfAttentionModel, self).__init__()
        self.res_conn = res_conn
        self.score_fn = score_fn

        self.embed = nn.Embedding(n_words, n_embed, padding_idx=pad_idx)
        self.linear = nn.Linear(n_embed, 1)
        self.drop = nn.Dropout(p_drop)

    def forward(self, inputs, mask):
        # (batch, len, n_embed)
        query = self.drop(self.embed(inputs))
        key = self.drop(self.embed(inputs))
        value = self.drop(self.embed(inputs))

        # (batch, len, 1)
        attn_mask = mask.unsqueeze(2)
        attn_mask = attn_mask * attn_mask.transpose(1, 2)

        # (batch, len, len)
        if self.score_fn == 'dot':
            score = torch.bmm(query, key.transpose(1, 2))
        elif self.score_fn == 'cos':
            query_norm = query / (query.norm(dim=-1).unsqueeze(2) + 1e-5)
            key_norm = key / (key.norm(dim=-1).unsqueeze(2) + 1e-5)
            score = torch.bmm(query_norm, key_norm.transpose(1, 2))

        score = score.masked_fill(~attn_mask, -1e9)
        attn = F.softmax(score, dim=-1)

        # (batch, n_embed)
        h_self = torch.bmm(attn, value).sum(1).squeeze()

        # 处理residual connection的情况
        if self.res_conn:
            # (batch, len, 1)
            mask = mask.float().unsqueeze(2)
            # (batch, len, n_embed)
            h_avg = query * mask
            # (batch, n_embed)
            h_avg = h_avg.sum(1) / (mask.sum(1) + 1e-5)
            h_self = h_self + h_avg

        return self.linear(self.drop(h_self)).squeeze()


class InputEmbedding(nn.Module):
    def __init__(self, n_words, d_model):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(n_words, d_model, padding_idx=1)

    def forward(self, x):
        # (bs, len, d_model)
        return self.embed(x) * np.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_drop=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p_drop)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        # (max_len, 1)
        pos = torch.arange(0, max_len).unsqueeze(1)
        # 计算分母，通过先转换为log之后exp获得
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 这里使用register_buffer，使得pe不会被优化器更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (bs, len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, p_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        # 这里保证head的数量一定要被d_model整除
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        # 每一个head中Q,K,V的维度
        self.d_hidden = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # 将经过线性变换的数据拆分为多个head
        # (bs, len, n_heads, d_hidden)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_hidden)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_hidden)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_hidden)

        # (bs, n_head, len, d_hidden)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 这里开始计算Attention
        # (bs, n_head, len, len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_hidden)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == False, -1e9)
        scores = F.softmax(scores, dim=-1)
        # (bs, n_head, len, d_hidden)
        attn = torch.matmul(scores, v)

        # (bs, len, d_model)
        concat = attn.transpose(1, 2).reshape(bs, -1, self.d_model)
        concat = self.dropout(concat)
        return self.fc(concat)


class SimpleTransformerModel(nn.Module):
    def __init__(self, n_words, d_model, n_heads, p_drop=0.1,
                 use_pos=False):
        super(SimpleTransformerModel, self).__init__()
        self.use_pos = use_pos

        self.inp_emb = InputEmbedding(n_words, d_model)
        self.pos_enc = PositionalEncoding(d_model, p_drop)
        self.mulattn = MultiHeadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, inputs, mask):
        # (bs, len, d_model)
        x_embed = self.inp_emb(inputs)

        if self.use_pos:
            # (bs, len, d_model)
            x_embed = self.pos_enc(x_embed)

        # (bs, len, 1)
        attn_mask = mask.unsqueeze(2)

        # (bs, len, len)
        attn_mask = attn_mask * attn_mask.transpose(1, 2)
        # (bs, len, d_model)
        x_attn = self.mulattn(x_embed, x_embed, x_embed, attn_mask)

        x_attn = self.norm(x_attn + x_embed)

        # (batch, len, 1)
        mask = mask.float().unsqueeze(2)
        # (batch, len, d_model)
        x_attn = x_attn * mask
        # (batch, d_model)
        x_sum = x_attn.sum(1) / (mask.sum(1) + 1e-6)
        return self.linear(x_sum).squeeze()


class PyTorchTransformerModel(nn.Module):
    def __init__(self, n_words, d_model, n_heads, n_layers, p_drop=0.1):
        super(PyTorchTransformerModel, self).__init__()
        self.inp_emb = InputEmbedding(n_words, d_model)
        self.pos_enc = PositionalEncoding(d_model, p_drop)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads,
                dim_feedforward=d_model * 2,
                dropout=p_drop
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        self.linear = nn.Linear(d_model, 1)

    def forward(self, inputs, mask):
        # (bs, len, d_model)
        x_embed = self.inp_emb(inputs)
        # (bs, len, d_model)
        x_embed = self.pos_enc(x_embed)

        # (len, bs, d_model)
        # 注意src_key_padding_mask参数中 pad的位置为True
        x_enc = self.encoder(x_embed.transpose(0, 1), src_key_padding_mask=~mask)
        x_enc = x_enc.transpose(0, 1)

        # (batch, len, 1)
        mask = mask.float().unsqueeze(2)
        # (batch, len, d_model)
        x_enc = x_enc * mask
        # (batch, d_model)
        x_sum = x_enc.sum(1) / (mask.sum(1) + 1e-6)
        return self.linear(x_sum).squeeze()


############################################################
# 测试模型
############################################################
def test_model():
    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    corpus = SSTCorpus(data_path)
    train_iter, valid_iter, test_iter = corpus.get_iterators((256, 256, 256))
    model = PyTorchTransformerModel(
        n_words=len(corpus.text_field.vocab),
        d_model=128,
        n_heads=4,
        n_layers=1,
    )
    model.to(DEVICE)

    for batch in train_iter:
        inputs, lengths = batch.text
        mask = (inputs != corpus.get_padding_idx())
        outputs = model(inputs, mask)
        print(outputs.shape)
        break


if __name__ == '__main__':
    test_model()
