__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/9
import argparse
from ml_idiot.tester.NormalTester import NormalTester


class LRTester(NormalTester):
    def __init__(self, config):
        super(LRTester, self).__init__(config)
