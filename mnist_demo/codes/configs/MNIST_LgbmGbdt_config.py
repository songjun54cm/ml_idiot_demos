__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/9
config = {
    'data_set_name': 'MNIST',
    'model_name': 'LR',
    "tester": "LRTester",
    "evaluator": "LREvaluator",
    'model_config': {
        "num_classes": 10,
        "num_leaves": 20,
        "num_trees": 100,
        "objective": "multiclass",
        "metric": ["multi_error", "multi_logloss"],
        "num_round": 10
    },
}
