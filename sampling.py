# -*- coding: utf-8 -*-
"""
__title__ = 'sampling'
__author__ = 'jz_hu'
__date__ = '2017/12/3'
"""
import copy
import numpy as np


class Sampling(object):
    def __init__(self, df_x, df_y):
        self.df_x = copy.deepcopy(df_x)
        self.df_y = copy.deepcopy(df_y)
        self.time_sampling_weight = tuple()
        self.quantity_sampling_weight = dict()

    def time_sampling(self, weight):
        """
        assume the data in dataframe range in [0, n-1] or time sequence
        :param weight: tuple (method, params) now method is only supported power distribution
                        params is the parameter of the distribution. Should be greater than zero.
                        set a=param
                        f(x)=ax^(a-1), 0<=x<=1, a>0
                            a>2:        f'>0, f">0
                            a=2: f=2x  a straight line
                            1<a<2:      f'>0, f"<0
                            a=1:       uniform distribution
                            0<a<1:      f'<0, f">0
        :return:
        """
        if not isinstance(weight, tuple) or len(weight) != 2 or not weight[0] is "power" or \
                isinstance(weight[1], (float, int)):
            raise ValueError("Weight type, shape, or value error: {}".format(weight))
        self.time_sampling_weight = weight
        s = np.random.power(weight[1], len(self.df_y))
        index = np.floor(s)
        return self.df_x.iloc[index, :], self.df_y.iloc[index, :]

    def quantity_sampling(self, weight):
        pass
