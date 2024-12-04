from __future__ import print_function,unicode_literals

import numpy as np
from numpy.ma.core import zeros_like
from scipy.spatial.transform import Slerp
from torch.nn.init import zeros_

from tools import *
from tools.quantilize import Discretor
from math import pi
from collections import Counter

epsilon = 1e-15

class Static:
    """统计分布类基类"""
    # 计算给定x的对数概率
    def predict_log_proba(self, x):
        p = self.predict_proba(x)
        log_P = np.log(p + epsilon)
        return log_P

    # 计算联合对数概率
    def joint_log_proba(self, x):
        log_p = self.predict_log_proba(x)
        return np.sum(log_p, axis=1)

class Normal(Static):
    """正态分布类"""
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, x, axis=0): #对每个维度独立统计参数
        self.mean = np.mean(x, axis=axis, keepdims=True)
        self.std = np.std(x, axis=axis, keepdims=True)

    def predict_prob(self, x):
        a = 1 / (np.sqrt(2*pi) * self.std)
        p = np.exp(-(x-self.mean)**2 / (2 * self.std**2))
        return a * p

class Multinomial(Static):
    """多项式分布类"""
    def __init__(self):
        self.pdfs = []

    def fit(self, x):
        n_samples, n_features = x.shape
        for j in range(n_features):
            xj = x[:,j]
            count = Counter(xj)
            pdf = {}
            for k,v in count.items():
                pdf[k] = float(v) / n_samples
            self.pdfs.append(pdf)

    def pdf_lookup(self, pdf):
        def map_fun(val):
            if val in pdf:
                return pdf[val]
            else:
                return 0

        return map_fun

    def predict_proba(self, x):
        n_samples, n_features = x.shape
        proba = np,zeros_like(x)
        for j in range(n_features):
            pdf = self.pdfs[j]
            proba[:,j] = np.vectorize(self.pdf_lookup(pdf))(x[:,j])

        return proba

class NaiveBayes:
    """朴素贝叶斯类
    参数：
        pdf: 似然函数分布类型
        discrete_level: 最大离散量化级别，如果是多项分布则用此参数
    """
    def __init__(self, pdf='gaussian', discrete_level=3):
        self.pdf = pdf
        if pdf == 'multinomial':
            self.discretor = Discretor(min_number=discrete_level)

    def fit(self, x, y):
        x, y = check_data(x, y)
        self.classes_ = np.unique(y)
        n_samples = len(y)
        self.log_priors = np.zeros(len(self.classes_))
        if self.pdf == 'gaussian':
            self.statics = [Normal() for _ in range(len(self.classes_))]
        elif self.pdf == 'multinomial':
            self.discretor.fit(x, y)
            x = self.discretor.predict(x)
            self.statics = [Multinomial() for _ in range(len(self.classes_))]

        for i,c in enumerate(self.classes_):
            c_x = x[y == c, :]
            self.statics[i].fit(c_x)
            self.log_priors[i] = np.log(len(c_x)) - np.log(n_samples)

    def predict_log_proba(self, x):
        x = np.asarray(x)
        if self.pdf == 'multinomial':
            x = self.discretor.predict(x)
        log_like = np.zeros((x.shape[0], len(self.classes_)))  # 似然概率
        for j,c in enumerate(self.classes_):
            log_like[:,j] = self.statics[j].joint_log_proba(x)
        log_post = log_like + self.log_priors
        return log_post

    def predict_proba(self, x):
        log_p = self.predict_log_proba(x)
        p = np.exp(log_p)
        return p / (np.sum(p, axis=1, keepdims=True) + epsilon)

    def predict(self, x):
        p = self.predict_proba(x)
        return self.classes_[p.argmax(1)]