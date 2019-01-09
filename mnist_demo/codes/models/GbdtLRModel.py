__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/9
import argparse
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import numpy as np
from ml_idiot.ml_models.NormalModel import NormalModel


class GbdtLRModel(NormalModel):
    def __init__(self, config):
        super(GbdtLRModel, self).__init__(config)
        self.num_round = 100
        self.gbdt = None
        self.lr = LogisticRegression()
        self.gbdt_params = {}
        self.lr_params = {}

    def create(self, config):
        self.gbdt_params["num_leaves"] = config.get("num_leaves", 31)
        self.gbdt_params["num_trees"] = config.get("num_trees", 100)
        self.gbdt_params["objective"] = config.get("objective", "multiclass")
        self.gbdt_params["metric"] = config.get("metric", ["multi_error", "multi_logloss"])
        self.gbdt_params["num_class"] = config.get("num_classes")
        self.num_round = config.get("num_round", 10)
        self.lr_params["solver"] = config.get("solver", "lbfgs")
        self.lr_params["multi_class"] = config.get("multi_class", "multinomial")
        self.lr_params["max_iter"] = config.get("max_iter", 100)

    def get_dataset(self, data):
        x = data["features"]
        y = data["labels"]
        dataset = lgb.Dataset(x, label=y)
        return dataset

    def train(self, train_data, valid_data=None, test_data=None):
        train_dataset = self.get_dataset(train_data)
        if valid_data is not None:
            valid_dataset = self.get_dataset(valid_data)
            self.gbdt = lgb.train(self.gbdt_params, train_dataset, self.num_round, valid_sets=[valid_dataset])
        else:
            self.gbdt = lgb.train(self.gbdt_params, train_dataset, self.num_round)

        x = train_data["features"]
        y = train_data["labels"]
        trans_features = self.transform_feature(x)
        self.lr = LogisticRegression(solver=self.lr_params["solver"],
                                     multi_class=self.lr_params["multi_class"],
                                     max_iter=self.lr_params["max_iter"])
        self.lr.fit(trans_features, y)

    def transform_feature(self, x):
        y_node = self.gbdt.predict(x, pred_leaf=True)
        num_trees = len(y_node[0])
        print(str(y_node.shape))
        trans_features = np.zeros((x.shape[0], num_trees * self.gbdt_params["num_leaves"]), dtype=np.int32)
        for i in range(x.shape[0]):
            idxs = np.arange(num_trees) * self.gbdt_params["num_leaves"] + np.array(y_node[i])
            trans_features[i][idxs] += 1
        return trans_features

    def predict_batch(self, batch_data):
        x = batch_data["features"]
        y = batch_data["labels"]
        trans_x = self.transform_feature(x)
        pred_vals = self.lr.predict(trans_x)
        res = {
            "loss": 0.0,
            "pred_vals": pred_vals,
            "gth_vals": y,
            "batch_size": x.shape[0]
        }
        return res