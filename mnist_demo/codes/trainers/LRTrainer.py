__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/9
import argparse
from ml_idiot.trainer.IterationTrainer import IterationTrainer


class LRTrainer(IterationTrainer):
    def __init__(self, config):
        super(LRTrainer, self).__init__(config)
