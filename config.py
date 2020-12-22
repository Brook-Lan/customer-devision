#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2020-12-16

@author:LHQ
"""
import os
from os.path import join as path_join


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")


class Config:
    ORIGIN_DATA_PATH = path_join(DATA_DIR, "fake_data.csv")     # 原始数据路径
    FAKE_DATA_PATH = path_join(DATA_DIR, "fake_data.csv")     # 伪数据文件路径
    CONTOUR_RESULT_DIR = path_join(DATA_DIR, "contour_result")      # 由原始数据生成图片的目录

    # 模型训练相关参数
    MODEL_NAME = "ConvAutoEncoder"
    IMG_DIR = path_join(DATA_DIR, "contour_result/img") 
    IMG_RESIZE = 160    # 输入模型的图片的大小设置 IMG_RESIZE * IMG_RESIZE
    VAL_SIZE = 0.3
    EPOCHS = 10
    BATCH_SIZE = 30
    LR = 0.0007
    MODEL_PATH = path_join(ROOT_DIR, "checkpoints")

