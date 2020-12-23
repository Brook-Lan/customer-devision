#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2020-12-09

@author:LHQ
"""
import os
from itertools import product

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def draw_corr_hotmap(df_corr, fig_path):
    """ 特征相关性热图
    Args
    ----
    df_corr : pandas.DataFrame 实例， columns和index均为特征名称， data为
        特征之间的相关系数
    fig_path : str, 生成的热图的保存位置

    Returns
    -------
    None
    """
    # 相关性热图
    plt.subplots(figsize=(9, 9))
    fig = sns.heatmap(df_corr, annot=True, vmax=1, square=True, cmap="Blues")
    plt.show()
    fig.get_figure().savefig(fig_path)
    plt.close()


def locate_feat(df_corr, scale=0.6):
    """ 定位每个特征在平面上的位置
    Args
    ----
    df_corr : pandas.DataFrame 实例， columns和index均为特征名称， data为
        特征之间的相关系数
    scale ： float， 坐标的范围约束值

    Returns
    -------
    G : nx.Graph, 特征节点图
    locations : dict , 每个特征的坐标(x,y)
    """

    # 根据相关性构造两两特征之前的边的权重(相关系数),进一步构造图
    index = df_corr.index
    columns = df_corr.columns
    pairs = product(index, columns)
    pairs = map(lambda p: tuple(sorted(p)), pairs)
    pairs = filter(lambda p: p[0] != p[1], set(pairs))
    edges = [(i, j, df_corr.loc[i, j]) for (i, j) in pairs]
    # build graph
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    locations = nx.spring_layout(G, k=0.1, iterations=500, scale=scale)
    return G, locations


def prepare_binning_args(df, bin_num=10):
    """准备数据分箱参数
    """
    # 大于0的部分分bin_num - 1 个箱
    data = []
    for feat in df.columns:
        tmp_s = df.query("%s > 0" % feat)[feat]    # 取大于0的比例值(0的数据比较多)
        data.append(tmp_s)
    _, bin_boundary = pd.qcut(pd.concat(data), q=bin_num-1, retbins=True)

    # 补充0值的分箱
    bin_boundary = [0] + list(bin_boundary)  # 起始补0
    values = [] 
    pre_v = None
    for i, v in enumerate(bin_boundary):
        if i > 1:
            median_v = (pre_v + v) / 2.0   # 取中间值作为该分箱对应的值
            values.append(median_v)
        elif i == 1:
            values.append(0)
        else:  # 等于0 则不操作
            pass
        pre_v = v

    label2value = {i:v for i, v in enumerate(values)}
    labels = list(range(len(values)))
    return bin_boundary, labels, label2value


def bining(df, bin_boundary, labels=False):
    """数据分箱
    """
    tmp = []
    for feat in df.columns:
        tmp.append(pd.cut(df[feat], bins=bin_boundary, labels=labels, include_lowest=True))
    df_bins = pd.concat(tmp, axis=1)
    return df_bins


def fill_height(x, y, h):
    """ 给定一个点的坐标(x, y)和高度h，构造一个山峰类填充该点周围的高度。
    将该山峰视为底圆直径为h，高度为h的圆锥，底圆圆心为(x,y)
    """
    heights = []
    half_h = int(h / 2)
    for i in range(x-half_h, x+half_h):
        for j in range(y-half_h, y+half_h):
            r = ((i - x) ** 2 + (j - y) ** 2) ** 0.5
            if r > half_h:
                continue
            h_ij = h - 2 * r
            heights.append((i, j, h_ij))
    return heights

    
def build_contour_img(data_path, result_dir, layout_scale=0.3, expand=100):
    """ 构建等高线图
    Args
    ----
    data_path : str, csv文件路径，columns包含 f_ + 数字 格式的特征名
    result_dir : str, 生成图片的保存目录
    layout_scale : float, 用于在spring layout时，限定节点在平面的坐标范围[-layout, layot]
    expand : int, 坐标、节点大小的放大倍数，放大后的数据用于构建等高线图
    """
    # 定义结果保存的文件目录与路径
    corr_fig_path = os.path.join(result_dir, "coor.jpg")
    relation_fig_path = os.path.join(result_dir, "relation.jpg")
    cust_bin_codes_file = os.path.join(result_dir, "cust_bin_codes.csv")
    img_dir = os.path.join(result_dir, "img")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    SCALE = layout_scale
    TIMES = expand

    ## 1.数据标准化
    df = pd.read_csv(data_path)
    feat_columns = ["f_%d" % i for i in range(11)]
    cust_total_v = df[feat_columns].sum(axis=1)  # 每个用户的数据总和
    df_p = df[feat_columns] / cust_total_v.values.reshape(-1, 1)   # 占比
    # 求整体的占比
    total_v = df[feat_columns].sum(axis=0)
    total_p = total_v / total_v.sum()

    df_correlation = df_p.corr().applymap(lambda v: round(v, 6))     # 特征的相关性

    ## 2.画相关性热图
    draw_corr_hotmap(df_correlation, corr_fig_path)
    ## 3.定位特征在平面上的坐标
    G, pos = locate_feat(df_correlation, SCALE)
    pos_new = pd.DataFrame(pos, index=["x", "y"])  \
             .applymap(lambda v: round(v, 3))  \
             .to_dict()    

    # 画节点-边 关联关系图
    nx.draw_networkx(G, 
        pos=pos,
        with_labels=True,
        width=0.5,
        nodelist=feat_columns,
        node_size=total_p*2000)
    plt.savefig(relation_fig_path)
    plt.close()
    ## 4.构造等高线地形图
    # 4.1 数据分箱
    bin_boundary, labels, label2value = prepare_binning_args(df_p, bin_num=10)
    df_bins = bining(df_p, bin_boundary, labels=labels)
    codes = []
    for i, row in df_bins.applymap(int).iterrows():
        code = "_".join(map(str,row.values))
        codes.append(code)
    df["bin_codes"] = codes
    df[["cust_id", "bin_codes"]].to_csv(cust_bin_codes_file, index=False)

    feats = df_bins.columns
    unique_codes = sorted(list(set(codes)))

    # 等高线图
    levels = np.linspace(min(bin_boundary), max(bin_boundary), 11) * TIMES
    half_axis_len = int(SCALE * 1.5 * TIMES)
    axis_len = half_axis_len * 2
    H = np.zeros((axis_len, axis_len))
    row = df_bins.iloc[0, :]
    ii = 0
    for code_str in unique_codes:
        bins = map(int, code_str.split("_"))
        for feat, bin_label in zip(feats, bins):
            wz = pos_new[feat]
            x, y = int(wz["x"] * TIMES + half_axis_len), int(wz["y"] * TIMES + half_axis_len)
            value = label2value[bin_label]    # 分箱对应的值
            heights = fill_height(x, y, TIMES * value)
            for xyh in heights:
                i, j, h = xyh
                if i < 0 or i >= axis_len or j < 0 or j >= axis_len:
                    continue
                H[i,j] = max(H[i,j], h)
        
        # 绘制等高线热图
        fig = plt.figure(figsize=(2, 2), dpi=100)
        ax = plt.Axes(fig, [0, 0, 1, 1])    # 画布边缘不留空白
        ax.set_axis_off()    # 关闭坐标轴
        fig.add_axes(ax)
        # ax.contourf(H, levels=9, alpha=0.9, cmap=plt.cm.hot)
        # ax.contourf(H, levels=levels, alpha=0.9, cmap=plt.cm.gist_ncar)
        # ax.contourf(H, levels=levels, alpha=0.9, cmap=plt.cm.coolwarm)
        # ax.contourf(H, levels=levels, alpha=0.9, cmap=plt.cm.bwr)
        ax.contourf(H, levels=levels, alpha=0.9, cmap=plt.cm.OrRd)
        img_path = os.path.join(img_dir, "%s.jpg" % code_str)
        fig.savefig(img_path, transparent=True)
        plt.close()
        ii += 1
        if ii > 5:
            pass
            # break
