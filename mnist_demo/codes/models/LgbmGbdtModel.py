__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/9
import argparse
import lightgbm as lgb
import numpy as np
from ml_idiot.ml_models.NormalModel import NormalModel


class LgbmGbdtModel(NormalModel):
    def __init__(self, config):
        super(LgbmGbdtModel, self).__init__(config)
        self.num_round = 0
        self.gbdt = None

    def create(self, config):
        self.params["num_leaves"] = config.get("num_leaves", 31)
        self.params["num_trees"] = config.get("num_trees", 100)
        self.params["objective"] = config.get("objective", "multiclass")
        self.params["metric"] = config.get("metric", ["multi_error", "multi_logloss"])
        self.params["num_class"] = config.get("num_classes")
        self.num_round = config.get("num_round", 10)

    def train(self, train_data, valid_data=None, test_data=None):
        train_dataset = self.get_dataset(train_data)
        if valid_data is not None:
            valid_dataset = self.get_dataset(valid_data)
            self.gbdt = lgb.train(self.params, train_dataset, self.num_round, valid_sets=[valid_dataset])
        else:
            self.gbdt = lgb.train(self.params, train_dataset, self.num_round)

    def get_dataset(self, data):
        x = data["features"]
        y = data["labels"]
        dataset = lgb.Dataset(x, label=y)
        return dataset

    def predict_batch(self, batch_data):
        x = batch_data["features"]
        y = batch_data["labels"]
        pred_probs = self.gbdt.predict(x)
        pred_vals = np.argmax(pred_probs, axis=1)
        res = {
            "loss": 0.0,
            "pred_vals": pred_vals,
            "gth_vals": y,
            "batch_size": x.shape[0]
        }
        return res

