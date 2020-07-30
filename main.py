#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @创建日期   :   2020/7/30 21:13
# @AUTHOR  :   梁开孟

import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

"""获取数据"""
train_url = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
df_iris_train = pd.read_csv(train_path, header=0)

test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)
df_iris_test = pd.read_csv(test_path, header=0)

"""将DataFrame转为array"""
iris_train = df_iris_train.values
iris_test = df_iris_test.values



