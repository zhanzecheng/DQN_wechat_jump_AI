# -*- coding: utf-8 -*-
"""
# @Time    : 2018/1/28 下午11:02
# @Author  : zhanzecheng
# @File    : ios_env.py
# @Software: PyCharm
"""
import wda
import time
import random
from env_base import Env_base

class IOS(Env_base):
    def __init__(self):
        self.c = wda.Client('http://localhost:8100')
        self.s = self.c.session()

    def pull_screenshot(self):
        self.c.screenshot('1.png')

    def tap_hold(self, action, loca = 85, locvb = 500):
        self.s.tap_hold(loca, locvb, action)

    def restart(self):
        self.s.tap(185, 500)
        time.sleep(random.uniform(1, 1.1))
