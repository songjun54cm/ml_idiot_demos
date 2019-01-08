__author__ = 'JunSong<songjun54cm@gmail.com>'
# Date: 2019/1/7
config = {
    'data_set_name': 'iris',
    'model_name': 'LgbmGbdt',
    'model_config': {
        "num_leaves": 20,
        "num_trees": 100,
        "objective": "multiclass",
        "metric": ["multi_error", "multi_logloss"],
        "num_round": 10
    },
}