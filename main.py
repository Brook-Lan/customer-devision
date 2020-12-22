#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2020-12-16

@author:LHQ
"""
import os

import click
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms as T

import models
from dataset import ContourDataset, make_transform
from config import Config
from utils import create_model_path, normalize_transpose


# models pkg下已有的模型
model_choices = click.Choice(["ConvAutoEncoder", "ConvAutoEncoderV2"])


@click.group()
def cli():
    pass


@cli.command()
@click.option("-o", "--output", type=click.File(mode="w"), default=Config.FAKE_DATA_PATH, help="生成的伪数据的保存位置")
def fake_data(output):
    """生成伪数据集(处理后可生成等高线图)
    """
    from sklearn.datasets import make_blobs

    N = 5 * 10000   # 数据量
    n_features = 11
    X, y = make_blobs(n_samples=N, n_features=n_features, centers=800, cluster_std=5)
    min_v = X.min(axis=0)
    X -= min_v
    df = pd.DataFrame(X, columns=["f_%d"%i for i in range(n_features)])
    df = df.applymap(lambda v: round(v, 2))
    df.insert(0, "cust_id", range(df.shape[0]))
    df.to_csv(output, index=False)


@cli.command()
@click.option("--data-path", type=click.File(), default=Config.ORIGIN_DATA_PATH, help="用户数据")
@click.option("--result-dir", type=click.Path(), default=Config.CONTOUR_RESULT_DIR, help="生成等高线图的保存目录")
def build_contour(data_path, result_dir):
    """ 处理数据，生成等高线图
    """
    from contour import build_contour_img

    build_contour_img(data_path, result_dir)


@cli.command()
@click.option("--model-name", type=model_choices, default="ConvAutoEncoder", help="模型名称")
@click.option("--load-model-path", type=click.File("rb"), help="加载该路径的下模型来进行后续的模型训练")
@click.option("--img-dir", type=click.Path(exists=True, file_okay=False), default=Config.IMG_DIR, help="用于模型训练的图片所在目录")
@click.option("--img-resize", type=click.INT, default=Config.IMG_RESIZE, help="图片大小调整参数")
@click.option("--val-size", type=click.FLOAT, default=Config.VAL_SIZE, help="验证集比例，取值0 ~ 1.0")
@click.option("--lr", type=click.FLOAT, default=Config.LR, help="学习率")
@click.option("--weight-decay", type=click.FLOAT, default=1e-5, help="")
@click.option("--epochs", type=click.INT, default=Config.EPOCHS)
@click.option("--batch-size", type=click.INT, default=Config.BATCH_SIZE)
@click.option("--model-dir", type=click.Path(file_okay=False), default=Config.MODEL_PATH, help="模型保存位置")
def train(
    model_name,
    load_model_path,
    img_dir,
    img_resize,
    val_size,
    lr,
    weight_decay,
    epochs,
    batch_size,
    model_dir
):
    """自编码器模型训练
    """

    from torch import optim
    import torch.nn as nn

    # step1: configure model
    model = getattr(models, model_name)()
    if load_model_path:
        model.load(load_model_path)

    # step2: prepare data
    train_data = ContourDataset(img_dir, resize=img_resize, test_size=val_size)
    val_data= ContourDataset(img_dir, resize=img_resize, test=True, test_size=val_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # step3: optimizer and loss func
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    # step4: train
    for epoch in range(epochs):
        train_num, val_num = 0, 0
        train_loss_epoch, val_loss_epoch = 0, 0
        model.train(True)
        for step, (data, img_name) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_func(outputs, data)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * data.size(0)
            train_num += data.size(0)

        model_path = create_model_path(model_dir, model_name)
        model.save(model_path)

        model.train(False)
        for step, (XX, img_name) in enumerate(val_loader):
            outputs = model(XX)
            loss = loss_func(outputs, XX)
            val_loss_epoch += loss.item() * XX.size(0)
            val_num += XX.size(0)

        train_loss = train_loss_epoch / train_num
        val_loss = val_loss_epoch / val_num
        print("epoch:{} train-loss:{:7f}  val-loss:{:7f}".format(epoch, train_loss, val_loss)) 


@cli.command()
@click.option("--model-name", type=model_choices, default="ConvAutoEncoder", help="模型名称")
@click.option("--load-model-path", type=click.File("rb"), help="模型文件位置")
@click.option("--img-path", type=click.File("rb"), help="图片位置")
@click.option("--img-resize", type=click.INT, default=Config.IMG_RESIZE, help="图片大小调整参数")
@click.option("--output-path", type=click.File("wb"), default="concat.jpg", help="输出图片文件路径")
def infer(
    model_name,
    load_model_path,
    img_path,
    img_resize,
    output_path
):
    # step1: load model
    model = getattr(models, model_name)()
    model.load(load_model_path)
    model.train(False)

    # step2: read img
    img = Image.open(img_path)

    transform = make_transform(resize=img_resize)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    # step3: infer
    output = model(img_tensor)

    # step4: conver to img
    output = normalize_transpose(output[0], mean=[0.485, 0.456,0.406], std=[0.229,0.224,0.225])
    img_new = T.ToPILImage()(output)

    img_old = normalize_transpose(img_tensor[0], mean=[0.485, 0.456,0.406], std=[0.229,0.224,0.225])
    img_old = T.ToPILImage()(img_old)
    
    width, height = img_old.size

    img_both = Image.new("RGB", (width * 2 + 30, height), "white")
    img_both.paste(img_old, (0,0))
    img_both.paste(img_new, (width + 30, 0))
    # draw = ImageDraw.Draw(img_both)
    # draw.text((40,40), "origin", fill="#000000")
    # draw.text((40,40), "auto_encode", fill="#000000")

    img_both.save(output_path)


@cli.command()
@click.option("--model-name", type=model_choices, default="ConvAutoEncoder", help="模型名称")
@click.option("--load-model-path", type=click.File("rb"), help="模型文件位置")
@click.option("--img-dir", type=click.Path(exists=True, file_okay=False), default=Config.IMG_DIR, help="用于模型训练的图片所在目录")
@click.option("--img-resize", type=click.INT, default=Config.IMG_RESIZE, help="图片大小调整参数")
@click.option("--batch-size", type=click.INT, default=Config.BATCH_SIZE)
@click.option("--output", type=click.File("w"), default="cluster_result.csv",  help="聚类结果")
def cluster(
    model_name,
    load_model_path,
    img_dir,
    img_resize,
    batch_size,
    output,
):
    from itertools import groupby
    from sklearn.cluster import DBSCAN
    import numpy as np

    # step1: load model
    model = getattr(models, model_name)()
    model.load(load_model_path)
    model.train(False)

    # step2: load data
    data = ContourDataset(img_dir, resize=img_resize)
    data_loader = DataLoader(data, batch_size=batch_size)

    # step3: encode
    fnames = []
    encodes = []
    for i, (img_arr, img_name) in enumerate(data_loader):
        encode = model(img_arr, output_encode=True)
        b = encode.size(0)
        encode = encode.view(b, -1).detach().numpy()
        encodes.append(encode)
        fnames.extend(img_name)
        if i > 30:
            break
    encodes = np.vstack(encodes)
    # step4: cluster
    y_pred_ = DBSCAN(eps=0.1, min_samples=3).fit_predict(encodes)
    y_pred = filter(lambda x: x[-1] > -1, enumerate(y_pred_))   ## 过滤异常点
    y_pred = sorted(map(lambda x: (fnames[x[0]], x[1]), y_pred), key=lambda x:x[-1])

    df_pred = pd.DataFrame(y_pred, columns=["fname", "group"])
    df_pred.to_csv(output, index=False)


    # groups = {}
    # for k, gp in groupby(y_pred, key=lambda x:x[1]):
    #     groups[k] = list(map(lambda x: x[0], gp))


    

if __name__ == "__main__":
    cli()
