# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 下午11:00
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : nlp-practice
# @File    : main.py
# @Desc    : Bert预训练模型


import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtext
import pytorch_lightning as pl
import transformers
from transformers import (
    DataProcessor,
    InputExample,
    BertForSequenceClassification,
    BertTokenizer,
    glue_convert_examples_to_features,
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SstProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir / 'senti.train.tsv'),
            set_type='train',
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir / 'senti.dev.tsv'),
            set_type='valid',
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir / 'senti.test.tsv'),
            set_type='test'
        )

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = f'{set_type}-{i}'  # 样本的唯一编号
            text_a = line[0]  # 预训练模型中的第一句话，因为是分类问题不需要第二句话
            label = line[1]  # 样本标签
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=None,
                label=label,
            ))
        return examples


def generate_dataloaders(tokenizer, data_path):
    def generate_dataloader_inner(examples):
        features = glue_convert_examples_to_features(
            train_examples,
            tokenizer,
            label_list=['0', '1'],
            max_length=128,
            output_mode='classification',
            pad_on_left=False,
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=0)

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor([f.input_ids for f in features]),
            torch.LongTensor([f.attention_mask for f in features]),
            torch.LongTensor([f.token_type_ids for f in features]),
            torch.LongTensor([f.label for f in features])
        )

        sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler=sampler, batch_size=32
        )
        return dataloader

    # 训练数据
    train_examples = processor.get_train_examples(data_path)
    train_loader = generate_dataloader_inner(train_examples)

    # 验证数据
    valid_examples = processor.get_dev_examples(data_path)
    valid_loader = generate_dataloader_inner(valid_examples)

    # 测试数据
    test_examples = processor.get_test_examples(data_path)
    test_loader = generate_dataloader_inner(test_examples)

    return train_loader, valid_loader, test_loader


class SstPreTrainedModel(pl.LightningModule):

    def __init__(self, model_path, train_loader, valid_loader, test_loader):
        super(SstPreTrainedModel, self).__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # 预训练模型
        self.ptm = BertForSequenceClassification.from_pretrained(
            model_path / 'bert-base-uncased' / 'bert-base-uncased-pytorch_model.bin',
            config=model_path / 'bert-base-uncased' / 'bert-base-uncased-config.json'
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.ptm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, label = batch
        out = self(input_ids, attention_mask, token_type_ids)

        loss = self.criterion(out, label)

        _, pred = torch.max(out, dim=1)
        acc = (pred == label).float().mean()

        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, label = batch
        out = self(input_ids, attention_mask, token_type_ids)

        loss = self.criterion(out, label)

        _, pred = torch.max(out, dim=1)
        acc = (pred == label).float().mean()

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': val_loss, 'val_acc': val_acc}
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, label = batch
        out = self(input_ids, attention_mask, token_type_ids)

        _, pred = torch.max(out, dim=1)
        acc = (pred == label).float().mean()

        return {'test_acc': acc}

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'test_acc': test_acc}
        return {'test_acc': test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


if __name__ == '__main__':
    data_path = Path('/media/bnu/data/nlp-practice/sentiment-analysis/standford-sentiment-treebank')
    model_path = Path('/media/bnu/data/nlp-practice/transformers')

    processor = SstProcessor()

    tokenizer = BertTokenizer.from_pretrained(
        model_path / 'bert-base-uncased' / 'bert-base-uncased-vocab.txt'
    )
    train_loader, valid_loader, test_loader = generate_dataloaders(tokenizer, data_path)

    model = SstPreTrainedModel(model_path, train_loader, valid_loader, test_loader)
    trainer = pl.Trainer(
        max_epochs=1,
        val_check_interval=0.1,
        gpus=2,
        distributed_backend='dp',
        default_save_path='/media/bnu/data/pytorch-lightning-checkpoints/'
    )
    trainer.fit(model)
    trainer.test(model)
