# -*- coding: utf-8 -*-
"""
__title__ = 'feature_map'
__author__ = 'jz_hu'
__date__ = '2017/11/30'
"""
import pandas as pd
import copy
import numpy as np
from scipy.special import boxcox1p
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing.data import QuantileTransformer

__all__ = ["FeatureMap"]


class FeatureMap(object):
    def __init__(self, df):
        self.df = copy.deepcopy(df)
        self.onehot = None
        self.label_code = None
        self.col_label_dict = dict()
        self.min_max_scale = None
        self.max_abs_scale = None
        self.standard_scale = None
        self.robust_scale = None
        self.quantile_transform = None

    def log_map(self, col_need, col_replace=True):
        df_need = self.df[col_need]
        if col_replace:
            self.df[col_need] = df_need.apply(lambda x: np.log(x))
        else:
            col_need_extend = [col + "_log" for col in col_need]
            self.df[col_need_extend] = df_need.apply(lambda x: np.log(x))

    def box_cox_map(self, col_need, gamma=1.0, col_replace=True):
        """
        y = ((1+x)**gamma - 1) / gamma  if gamma != 0
            log(1+x)                    if gamma == 0
        ref: http://onlinestatbook.com/2/transformations/box-cox.html
        :param col_need:
        :param gamma:
        :param col_replace:
        :return:
        """
        df_need = self.df[col_need]
        if col_replace:
            self.df[col_need] = df_need.applymap(lambda x: boxcox1p(x, gamma))
        else:
            col_need_extend = [col + "_boxCox" for col in col_need]
            self.df[col_need_extend] = df_need.applymap(lambda x: boxcox1p(x, gamma))

    def onehot_encode(self, col_need, start_zero=True):
        """
        onehot encode DataFrame of which the columns you need
        note: the origin category should be integer in range(classes) or range(classes+1)
        :param col_need:
        :param start_zero: category is in range(classes)
        :return: new DataFrame without col_need, after onehot encoding,
                  start method is in accordance with start_zero
        """
        self.onehot = OneHotEncoder(sparse=False)
        array_onehot = self.onehot.fit_transform(self.df.loc[:, col_need])

        col_onehot = []

        for col_index in range(len(col_need)):
            if start_zero:
                for hot_index in range(self.onehot.n_values_[col_index]):
                    col_onehot.append(col_need[col_index] + str(hot_index))
            else:
                for hot_index in range(1, self.onehot.n_values_[col_index]):
                    col_onehot.append(col_need[col_index] + str(hot_index))

        self.df.drop(col_need, axis=1, inplace=True)

        df_onehot = pd.DataFrame(array_onehot, columns=col_onehot, index=self.df.index)
        self.df = pd.concat([self.df, df_onehot], axis=1)

    def label_encode(self, col_need):
        """
        onehot encode DataFrame of which the columns you need
        :param col_need: length should be 1
        :return: new DataFrame without col_need, after label encoding, start from 0
        """
        assert isinstance(col_need, list) and len(col_need) == 1
        self.label_code = LabelEncoder()
        array_label_code = self.label_code.fit_transform(self.df.loc[:, col_need])

        label_list = list(self.label_code.classes_)
        for i, x in enumerate(label_list):
            self.col_label_dict[col_need[0] + "_" + str(i)] = col_need[0] + "_" + x

        self.df.drop(col_need, axis=1, inplace=True)

        df_label_code = pd.DataFrame(array_label_code, columns=col_need, index=self.df.index)
        self.df = pd.concat([self.df, df_label_code], axis=1)

    def standard_scale_map(self, col_need, drop_origin_col=False):
        self.standard_scale = StandardScaler()
        array_standard = self.standard_scale.fit_transform(self.df.loc[:, col_need])
        self._scale_map(array=array_standard,
                        column_name=col_need,
                        suffix="_stdScale",
                        drop_origin_columns=drop_origin_col)

    def min_max_scale_map(self, col_need, drop_origin_col=False):
        self.min_max_scale = MinMaxScaler()
        array_min_max = self.min_max_scale.fit_transform(self.df.loc[:, col_need])
        self._scale_map(array=array_min_max,
                        column_name=col_need,
                        suffix="_minMaxScale",
                        drop_origin_columns=drop_origin_col)

    def max_abs_scale_map(self, col_need, drop_origin_col=False):
        self.max_abs_scale = MaxAbsScaler()
        array_max_abs = self.max_abs_scale.fit_transform(self.df.loc[:, col_need])
        self._scale_map(array=array_max_abs,
                        column_name=col_need,
                        suffix="_maxAbsScale",
                        drop_origin_columns=drop_origin_col)

    def robust_scale_map(self, col_need, quantile_range=(25, 75), drop_origin_col=False):
        """
        This Scaler removes the median and scales the data according to
        the quantile range (defaults to IQR: Interquartile Range).
        The IQR is the range between the 1st quartile (25th quantile)
        and the 3rd quartile (75th quantile).
        :param col_need:
        :param quantile_range:
        :param drop_origin_col:
        :return:
        """
        self.robust_scale = RobustScaler(quantile_range=quantile_range)
        array_robust = self.robust_scale.fit_transform(self.df.loc[:, col_need])
        self._scale_map(array=array_robust,
                        column_name=col_need,
                        suffix="_robust_scale",
                        drop_origin_columns=drop_origin_col)

    def quantile_scale_map(self, col_need, distribution='uniform', drop_origin_col=False):
        """

        :param col_need:
        :param distribution: 'uniform' (default) or 'normal'
        :param drop_origin_col:
        :return:
        """
        self.quantile_transform = QuantileTransformer(output_distribution=distribution)

        array_quantile = self.quantile_transform.fit_transform(self.df.loc[:, col_need])
        self._scale_map(array=array_quantile,
                        column_name=col_need,
                        suffix="_q{}Map".format(distribution.capitalize()),
                        drop_origin_columns=drop_origin_col)

    def _scale_map(self, array, column_name, suffix, drop_origin_columns=False):
        if drop_origin_columns:
            self.df.drop(column_name, axis=1, inplace=True)

        col = [col + suffix for col in column_name]
        df_scale = pd.DataFrame(array, columns=col, index=self.df.index)
        self.df = pd.concat([self.df, df_scale], axis=1)

    def quantile_floor_map(self, col_need, floor_num=5, drop_origin_col=False):
        """
        after quantile_scale_map when distribution='uniform', value is scaled in [0, 1]
        for tree models, onehot encoding is need
        :param col_need:
        :param floor_num: uniform floor map
        :param drop_origin_col
        :return:
        """
        bool0 = (self.df.loc[:, col_need] >= 0) & (self.df.loc[:, col_need] <= 1)
        assert bool0.all().all()
        col_suffix = np.array([x.endswith("_qUniformMap") for x in col_need])
        assert np.prod(col_suffix)

        array_quantile_floor = (self.df.loc[:, col_need].values * floor_num).astype(np.int)
        self._scale_map(array=array_quantile_floor,
                        column_name=col_need,
                        suffix="_qFloorMap",
                        drop_origin_columns=drop_origin_col)

"""
if __name__ == "__main__":
    # df0 = pd.read_csv("D:\\Users\jz_hu\Desktop\\x_df_im1.csv")
    # df_test = df0.loc[:, ["ordertotalbyorderdate", "hour", "week"]]
    # fp = FeatureMap(df_test)
    # fp.log_map(["ordertotalbyorderdate", "ordertotalbytakeofftime"], col_replace=True)
    # print(fp.df.head())
    # fp.box_cox_map(["ordertotalbyorderdate", "ordertotalbytakeofftime"], gamma=0.25)
    # print(fp.df.head())
    # fp.onehot_encode(["week"], start_zero=False)
    # print(fp.df.head())
    # fp.onehot_encode(["hour"], start_zero=True)
    # print(fp.df.head())
    # df1 = pd.DataFrame({"col1": ['a', 'b', 'c', 'a'],
    #                     "col2": ['x', 'y', 'x', 'y']
    #                     })
    # fp = FeatureMap(df1)
    # fp.label_encode(['col1'])
    # print(fp.df.head())
    # print(fp.col_label_dict)
    #
    # fp.label_encode(['col2'])
    # print(fp.df.head())
    # print(fp.col_label_dict)
    df2 = pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'])
    fp = FeatureMap(df2)
    #     fp.max_abs_scale_map(['a', 'b'])
    #     fp.standard_scale_map(['a', 'b'])
    #     fp.min_max_scale_map(['a', 'b'])
    #     fp.robust_scale_map(['a', 'b'])
    fp.quantile_scale_map(['a', 'b'], distribution="uniform")
    #     fp.quantile_scale_map(['a', 'b'], distribution="normal")
    print(fp.df.head())
    fp.quantile_transform.transform(np.random.randn(10, 2))

"""