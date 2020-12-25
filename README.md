# 客户微细分

## Table of Contents
- [Intro](#intro)
- [Install](#install)
- [Usage](#usage)


## Intro

对用户的特征(比如各项资产情况)做特征之间的相关性分析，据此布局特征的关系图，得到每个特征(资产类型)的空间坐标，然后以资产占比为高度，生成等高线地形图，然后通过训练自编码器，获取等高线地形图的编码，进一步做聚类，划分用户群


## Install

环境 python3.6+

```
pip3 install -r requirements.txt
```

## Usage

```
python3 main.py [OPTIONS] COMMAND [ARGS]...
```

--help 查看命令与参数介绍

```
python3 main.py --help
```
```
python3 main.py build-contour --help
```

1.生成伪数据

```
python3 main.py fake-data
```
执行该命令，将生成伪数据 `data/fake_data.csv`，若有外部的业务数据，则忽略这一步（但外部数据的格式需和`data/fake_data.csv`保持一致)

2.处理数据，生成等高线地形图(--help 查看该命令需要的参数)

```
python3 main.py build-contour
```

3.以第2步生成的等高线地形图为训练数据训练自编码器(--help 查看该命令需要的参数)

```
python3 main.py train
```

4.自编码器的效果展示(--help 查看该命令需要的参数)

```
python3 main.py infer 
```

5.聚类(--help 查看该命令需要的参数)

```
python3 main.py cluster
```