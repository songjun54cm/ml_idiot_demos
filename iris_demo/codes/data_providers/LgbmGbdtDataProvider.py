__author__ = 'JunSong<songjun54cm@gmail.com>'
# Date: 2019/1/7
import argparse
import logging
import random
import numpy as np
import lightgbm as lgb
from ml_idiot.data_provider.NormalDataProvider import NormalDataProvider


class LgbmGbdtDataProvider(NormalDataProvider):
    def __init__(self):
        super(LgbmGbdtDataProvider, self).__init__()
        self.split_ratios = [0.8, 0.1, 0.1]

    def load_raw_data_samples(self, config):
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        records = []
        for (a, b) in zip(X, y):
            records.append((a, b))
        labels = [x[1] for x in records]
        num_class = len(set(labels))
        config["model_config"]["num_classes"] = num_class
        return records

    # def form_dataset(self):
    #     x, y = self.get_split_dataset("train")
    #     train_dataset = lgb.Dataset(x, label=y)
    #     train_dataset.free_raw_data = False
    #     self.splits["train"] = train_dataset.construct()
    #
    #     x, y = self.get_split_dataset("train_valid")
    #     train_valid_dataset = lgb.Dataset(x, label=y)
    #     train_valid_dataset.free_raw_data = False
    #     self.splits["train_valid"] = train_valid_dataset.construct()
    #
    #     x, y = self.get_split_dataset("valid")
    #     valid_dataset = lgb.Dataset(x, label=y)
    #     valid_dataset.free_raw_data = False
    #     self.splits["valid"] = valid_dataset.construct()
    #
    #     x, y = self.get_split_dataset("test")
    #     test_dataset = lgb.Dataset(x, label=y)
    #     test_dataset.free_raw_data = False
    #     self.splits["test"] = test_dataset.construct()



    # def create(self, config):
    #     super(LgbmGbdtDataProvider, self).build(config)
    #     self.form_dataset()
    #     self.summarize()

    # def summarize(self):
    #     super(NormalDataProvider, self).summarize()
    #     logging.info('train data size: %d' % self.get_split('train').num_data())
    #     logging.info('valid data size: %d' % self.get_split('valid').num_data())
    #     logging.info('test data size: %d' % self.get_split('test').num_data())

    # def iter_split_batches(self, batch_size, split, rng=random.Random(1234), shuffle=True, mode='random', opts=None):
    #     if batch_size is None or batch_size <= 0:
    #         yield self.get_split(split)
    #     else:
    #         raise NotImplementedError

    def create(self, config):
        super(LgbmGbdtDataProvider, self).build(config)
        super(LgbmGbdtDataProvider, self).summarize()