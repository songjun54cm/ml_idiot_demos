__author__ = 'JunSong<songjun54cm@gmail.com>'
# Date: 2019/1/8
import argparse
from ml_idiot.evaluator.MultiClassifyEvaluator import MultiClassifyEvaluator


class LgbmGbdtEvaluator(MultiClassifyEvaluator):
    def __init__(self, config):
        super(LgbmGbdtEvaluator, self).__init__(config)
