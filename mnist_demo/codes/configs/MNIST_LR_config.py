__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse


config = {
    'data_set_name': 'MNIST',
    'model_name': 'LR',
    'batch_iter_n': 1,
    'max_epoch': 10,
    'model_config': {
        'learning_rate': 'constant',
        'eta0': 0.0005
    },
}