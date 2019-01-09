__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/9
import argparse
from ml_idiot.evaluator.MultiClassifyEvaluator import MultiClassifyEvaluator


class LREvaluator(MultiClassifyEvaluator):
    def __init__(self, config):
        super(LREvaluator, self).__init__(config)
