#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2020-12-17

@author:LHQ
"""
import torch as t


class BasicModule(t.nn.Module):
    """封装了nn.Module， 主要提供了save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))


    def load(self, path):
        """ 加载模型
        """
        self.load_state_dict(t.load(path))

    def save(self, path):
        """ 模型保存
        """
        t.save(self.state_dict(), path)
