__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/9
import argparse
from ml_idiot.trainer.NormalTrainer import NormalTrainer


class LgbmGbdtTrainer(NormalTrainer):
    def __init__(self, config):
        super(LgbmGbdtTrainer, self).__init__(config)
