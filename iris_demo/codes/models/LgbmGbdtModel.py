__author__ = 'JunSong<songjun54cm@gmail.com>'
# Date: 2019/1/7
import argparse
import numpy as np
import lightgbm as lgb
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

    def train_early_stop(self, train_data, valid_sets, early_stop_round=10):
        self.gbdt = lgb.train(self.params, train_data, self.num_round, valid_sets=valid_sets,
                              early_stopping_rounds=early_stop_round)
        # self.gbdt.save_model("model.txt", num_iteration=self.gbdt.best_iteration)

    def save(self, path):
        self.gbdt.save_model(path)

    def load(self, path):
        self.gbdt = lgb.Booster(model_file=path)

    def to_json(self):
        return self.gbdt.dump_model()

    def predict(self, in_data):
        return self.gbdt.predict(in_data)

    def predict_batch(self, batch_data):
        x, y = self.get_batch_data(batch_data)
        pred_probs = self.gbdt.predict(x)
        pred_vals = np.argmax(pred_probs, axis=1)
        res = {
            "loss": 0.0,
            "pred_vals": pred_vals,
            "gth_vals": y,
            "batch_size": len(batch_data)
        }
        return res

    def get_batch_data(self, batch_data):
        xs = [d[0] for d in batch_data]
        ys = [d[1] for d in batch_data]
        data_x = np.asarray(xs)
        data_y = np.asarray(ys)
        return data_x, data_y

    def get_dataset(self, data):
        x, y = self.get_batch_data(data)
        dataset = lgb.Dataset(x, label=y)
        # dataset.free_raw_data = False
        # dataset.construct()
        return dataset
