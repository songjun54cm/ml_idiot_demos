__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/9
import argparse
import numpy as np
import _pickle as pkl
from ml_idiot.data_provider.NormalDataProvider import NormalDataProvider


class LgbmGbdtDataProvider(NormalDataProvider):
    def __init__(self):
        super(LgbmGbdtDataProvider, self).__init__()

    def get_one_batch(self, split_datas, split_size, idxs, start_pos, end_pos):
        res = {
            "features": split_datas["features"][start_pos:end_pos,:],
            "labels": split_datas["labels"][start_pos:end_pos],
            "batch_size": min(split_size, end_pos) - start_pos
        }
        return res

    def build(self, config):
        data_file = "../data/mnist/mnist_data/mnist.pkl"
        with open(data_file, "rb") as f:
            mnist_data = pkl.load(f)
        train_images = mnist_data["train_images"]
        train_features = np.reshape(train_images, (train_images.shape[0], train_images.shape[1]*train_images.shape[2]))
        # max_fea = np.max(train_features)
        # min_fea = np.min(train_features)
        # temp = (train_features > 255*0.3).astype(np.int32),
        # max_t = np.max(temp)
        # min_t = np.min(temp)
        train_labels = mnist_data["train_labels"]
        self.splits["train"] = {
            "features": (train_features > 255*0.3).astype(np.int32),
            "labels": train_labels,
            "split_size": train_features.shape[0]
        }
        test_images = mnist_data["test_images"]
        test_features = np.reshape(test_images, (test_images.shape[0], test_images.shape[1]*test_images.shape[2]))
        test_labels = mnist_data["test_labels"]
        self.splits["test"] = {
            "features": (test_features > 255*0.3).astype(np.int32),
            "labels": test_labels,
            "split_size": test_features.shape[0]
        }