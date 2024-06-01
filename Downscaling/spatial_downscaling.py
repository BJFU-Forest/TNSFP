# coding=utf-8
import os
import random

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, neural_network
from PyEMD.visualisation import Visualisation
import joblib
from sklearn.feature_selection import SequentialFeatureSelector

from Tools.Evaluation.Index import NSE, get_rsr, PBIAS

import warnings

warnings.filterwarnings("ignore")


def data_split(X, y, date, train_size, group_size=10):
    """
    训练/测试集分组
    :param X: 因子集
    :type X: np.array or list
    :param y: 标签集
    :type y: np.array or list
    :param date: 被分组数据的时间列 ndarray[datetime]
    :type date: np.array or list
    :param train_size: 训练集占比
    :type train_size: float
    :param group_size: 采样分组大小(年) 默认为10年
    :type group_size: float
    :return:
    """

    # 检查数据格式
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    if not isinstance(date, np.ndarray):
        date = np.asarray(date)
    date = pd.to_datetime(date)

    # 判断group_size的设置是否合适
    years = set(date.year)
    group_size = group_size if len(years) >= group_size else len(years)

    # 获取训练/测试集分组索引
    train_year = {year for i, year in enumerate(years) if i % group_size < train_size * group_size}
    test_year = years ^ train_year
    train_index = date.year.isin(train_year)
    test_index = date.year.isin(test_year)

    # 训练/测试集分组
    X_train = X[train_index]
    X_test = X[test_index]
    Y_train = y[train_index]
    Y_test = y[test_index]
    return X_train, X_test, Y_train, Y_test


def scaler_handler(df, backupPath, name, scaler=None, labels=None):
    # 备份标签
    df_label = df.columns.values
    if (scaler is None) or (labels is None):
        # 备份缩放器
        scaler = preprocessing.MinMaxScaler()
        norm_data = scaler.fit_transform(df.values)
        joblib.dump(scaler, backupPath + name + "_min_max_scaler.pkl")
    else:
        new_df = pd.DataFrame(columns=labels, index=df.index, data=np.full([df.shape[0], labels.shape[0]], np.nan))
        col_sel = np.in1d(labels, df.columns.values)
        df_label = new_df.columns.values[col_sel]
        new_df[df_label] = df[df_label]
        norm_data = scaler.transform(new_df.values)[:, col_sel]
    np.save(backupPath + name + "_label.npy", df_label)
    return df_label, norm_data, scaler


class PredictorsDownscaling(object):
    """
    Statistical downscaling infers higher resolution information from lower resolution data.
    For example, data collected at a more coarse regional level applied to a more refined
    local level.

    Statistical downscaling establishes a relationship between different variables in the large scale
    and the local scale and applies that relationship to the local scale.

    Remove the biases by fitting a linear regression model with ordered observational and model datasets

        Hessami et al (2008) An regression-based downscaling methods model for statistical downscaling of
        daily climate variables

    Classes:
    Downscaling use the large scale atmospheric variables (or "predictors")
      - Container for applying automated regression-based statistical downscaling (Hessami et al., 2008).
      Predict:
        √ multiple linear regressions
        √ the ridge regression (Hoerl and Kennard, 1970)
      Select best predictors
        √ stepwise regression (forward, backwards)
        √ partial correlation
    """

    def __init__(self, ref_data, predictors, model_backup_path, train_size=0.8, group_size=10, max_predictors=None,
                 method="forward", is_condition=False,
                 ref_decomp=None, tor_decomp=None, tor_scaler=None, tor_labels=None, select_best=False):
        """ Default Downscaling constructor.

        :param ref_data: The Dataset to use as the reference dataset (observation)
        :type ref_data: pandas.DataFrame
        :param predictors: model simulation predictors to be compared with observation
        :type predictors: pandas.DataFrame
        :param model_backup_path: model backup path
        :type model_backup_path: str[path]
        :param train_size: Proportion of training data sets [0-1]
        :type train_size: float
        :param group_size: 采样分组大小(年) 默认为10年
        :type group_size: float
        :param max_predictors: 最多预测因子数
        :type max_predictors: int
        :param method: 预测因子筛选方法
        :type method: str
        :param is_condition: 是否条件拟合(>=0)
        :type is_condition: bool
        :param ref_decomp: The Dataset to use as the reference dataset (observation) after Decomposition
        :type ref_decomp: pandas.DataFrame
        :param tor_decomp: The Dataset to use as the predictor dataset after Decomposition
        :type tor_decomp: pandas.DataFrame
        """
        self.ref_dataset = ref_data
        self.predictors = predictors
        self.model_backup_path = model_backup_path
        self.train_size = train_size
        self.group_size = group_size
        self.max_predictors = max_predictors
        self.method = method
        self.is_condition = is_condition
        self.model_info = r" ".join(model_backup_path.split("/")[-3:-1])
        self.ref_decomp = ref_decomp
        self.tor_decomp = tor_decomp
        self.tor_scaler = tor_scaler
        self.tor_labels = tor_labels
        self.select_best = select_best

        # 建模输入
        self.y = ref_decomp if ref_decomp is not None else ref_data
        self.X = pd.concat([predictors, tor_decomp], axis=1) if tor_decomp is not None else predictors
        # 保存因子状态
        ref_state = True if ref_decomp is not None else False
        np.save(model_backup_path + "ref_state.npy", ref_state)
        # 预测因子标准化
        self.X_labels, self.X_norm, X_scaler = scaler_handler(self.X, model_backup_path, "X", tor_scaler, tor_labels)
        y_model = self.y.values
        # 被预测因子标准化
        y_labels, y_model, self.y_scaler = scaler_handler(self.y, model_backup_path, "y")
        # 预测因子索引
        self.X_index = [np.in1d(self.X_labels, predictors.columns.values)]
        if tor_decomp is not None:
            self.X_index.append(np.in1d(self.X_labels, tor_decomp.columns.values))
            self.X_index.append(np.full(self.X_labels.shape[0], True))

        # Decomposition_Y可视化
        if ref_decomp is not None:
            self.vis = Visualisation()
            decomp_fig = self.vis.plot_imfs(ref_decomp.values.T[:-1], residue=ref_decomp.values.T[-1], t=None,
                                            include_residue=True)
            decomp_fig.savefig(model_backup_path + self.model_info + " Y_decomp.jpg")

        # 训练数据分组
        self.X_train, self.X_test, self.y_train, self.y_test = data_split(
            self.X_norm, y_model, self.y.index,
            train_size=self.train_size, group_size=self.group_size)

    description = "statistical downscaling methods"

    def multiple_linear_regressions(self):
        # 模型参数导出
        predictors_dict = {}
        # 模型输出
        present_fitted = []

        """确定性部分（按全部数据降尺度）"""
        for i in range(self.y_train.shape[1]):
            # 预测因子筛选
            selector = SequentialFeatureSelector(linear_model.LinearRegression(),
                                                 n_features_to_select=self.max_predictors, direction=self.method,
                                                 scoring='neg_mean_squared_error',
                                                 cv=5, n_jobs=10)
            selector.fit(self.X_train, self.y_train)
            # 构建预测因子集
            x_index = selector.get_support()
            predictors = self.X_labels[x_index]
            Y_train = self.y_train[:, i]  # 训练因变量集
            X_train = self.X_train[:, x_index]  # 训练因子集
            X_norm = self.X_norm[:, x_index]  # 预测因子集

            # 拟合模型
            model = linear_model.LinearRegression(n_jobs=-1)
            model.fit(X_train, Y_train)
            # 保存模型
            joblib.dump(model, self.model_backup_path + "mlr_imf%d.pkl" % (i + 1))
            # 模型参数导出
            predictors_dict[i] = predictors.tolist()
            # 模型输出
            present_fitted.append(model.predict(X_norm))

        # 整合IMF预测结果
        present_fitted = np.asarray(present_fitted).sum(axis=0)

        # 正值条件判断
        present_fitted[present_fitted < 0] = 0 if self.is_condition else present_fitted[present_fitted < 0]

        # 参数储存
        np.save(self.model_backup_path + "mlr_param.npy", predictors_dict)
        # 参数输出
        predictors_dict_copy = predictors_dict.copy()
        for key in predictors_dict_copy.keys():
            predictors_dict_copy[key] = ",".join(predictors_dict_copy[key])
        mlr_param = pd.DataFrame(predictors_dict_copy, index=[0])
        mlr_param.to_csv(self.model_backup_path + "mlr_param.csv")

        return present_fitted

    def multilayer_perceptron(self):
        ###########################################内置函数#############################################################
        def _fit_model(X_train, y_train, X_test, y_test):
            """拟合多层感知机神经网络模型"""
            model = neural_network.MLPRegressor(
                hidden_layer_sizes=(500, 500, 250),
                activation="relu",
                alpha=0.0001,
                solver="adam",
                batch_size="auto",
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=100000,
                # early_stopping=True,
                # validation_fraction=0.4,
                tol=1e-4,
                n_iter_no_change=100,
                verbose=False,
                warm_start=False,
            )
            try:
                model.fit(X_train, y_train)
                cal_score = NSE(y_train, model.predict(X_train))
                val_score = NSE(y_test, model.predict(X_test))
                return model, cal_score, val_score
            except:
                return None, -np.inf, -np.inf

        def _get_IMF_index(imf_index):
            if imf_index == self.y.shape[1] - 1:
                IMF_index = np.asarray(
                    [True if "residual" in label else False for label in self.X_labels])
            else:
                IMF_index = np.asarray(
                    [True if "imf%d" % (imf_index + 1) in label else False for label in self.X_labels])
            return IMF_index

        def _select_best_model(y_train, y_test, index_list, backup_name):
            models = []
            for ind in index_list:
                if not ind.any():
                    models.append((None, -np.inf, -np.inf))
                    continue
                models.append(_fit_model(self.X_train[:, ind], y_train, self.X_test[:, ind], y_test))
            models = np.asarray(models)
            best_model = np.argmax(models[:, 2])
            model, cal_score, val_score = models[best_model]
            index = index_list[best_model]
            print("    Score: %s || %s >>>>>> % .4f vs. % .4f    Finished." % (
                " | ".join(np.char.mod("% .4f", models[:, 1])), " | ".join(np.char.mod("% .4f", models[:, 2])),
                cal_score,
                val_score))
            # 保存模型
            joblib.dump(model, self.model_backup_path + "mlp_y_" + backup_name + ".pkl")  # mlp_y_original.pkl
            # 保存预测因子标签
            np.save(self.model_backup_path + "X_" + backup_name + "_labels.npy",
                    self.X_labels[index])  # X_original_label.npy
            # 模型输出
            fitted = model.predict(self.X_norm[:, index])
            return model, index, fitted, best_model

        def _IMFs_model(y_index, y_train, y_test, index_list, backup_name):
            if y_index is None:
                index = index_list[0]
                model, cal_score, val_score = _fit_model(self.X_train[:, index], y_train, self.X_test[:, index], y_test)
            else:
                index = index_list[3]
                model, cal_score, val_score = _fit_model(self.X_train[:, index], y_train, self.X_test[:, index], y_test)
            print("    Score:  % .4f vs. % .4f    Finished." % (cal_score, val_score))
            # 保存模型
            joblib.dump(model, self.model_backup_path + "mlp_y_" + backup_name + ".pkl")  # mlp_y_original.pkl
            # 保存预测因子标签
            np.save(self.model_backup_path + "X_" + backup_name + "_labels.npy",
                    self.X_labels[index])  # X_original_label.npy
            # 模型输出
            fitted = model.predict(self.X_norm[:, index])
            return model, index, fitted

        ###########################################内置函数#############################################################
        if self.y.shape[1] == 1:
            print("      Fit model '%s mlp_y_original.pkl'" % self.model_info, end="")
            X_index = self.X_index.copy()
            if self.tor_decomp is not None:
                IMF_index = np.asarray([True if "imf2" in label else False for label in self.X_labels])
                X_index.append(IMF_index)
                X_index.append(np.logical_or(self.X_index[0], IMF_index))
            if self.select_best:
                model, index, present_fitted, _ = _select_best_model(self.y_train, self.y_test, X_index, "original")
            else:
                model, index, present_fitted = _IMFs_model(None, self.y_train, self.y_test, X_index, "original")
            present_fitted = self.y_scaler.inverse_transform(np.asarray(present_fitted).reshape(-1, 1)).reshape(-1)
        elif self.y.shape[1] > 1:
            # 模型输出
            present_fitted = []
            for i in range(self.y.shape[1]):
                print("      Fit IMF model '%s mlp_imf%d.pkl'" % (self.model_info, i + 1), end="")
                # 更新Xindex列表
                X_index = self.X_index.copy()
                if self.tor_decomp is not None:
                    IMF_index = _get_IMF_index(imf_index=i)
                    X_index.append(IMF_index)
                    X_index.append(np.logical_or(self.X_index[0], IMF_index))
                if self.select_best:
                    model, index, fitted, best_model = _select_best_model(self.y_train[:, i], self.y_test[:, i],
                                                                          X_index, "imf%d" % (i + 1))
                else:
                    model, index, fitted = _IMFs_model(i, self.y_train[:, i], self.y_test[:, i],
                                                       X_index, "imf%d" % (i + 1))
                # 模型输出
                present_fitted.append(fitted)
            # 绘制MLP模拟结果
            present_fitted = self.y_scaler.inverse_transform(np.asarray(present_fitted).T).T
            decomp_fig = self.vis.plot_imfs(imfs=np.asarray(present_fitted)[:-1],
                                            residue=np.asarray(present_fitted)[-1],
                                            t=None, include_residue=True)
            decomp_fig.savefig(self.model_backup_path + self.model_info + " Fitted.jpg")
            # 整合IMF预测结果
            present_fitted = np.asarray(present_fitted).sum(axis=0)

        else:
            raise Exception("y输入错误，y.shape[1]必须>=1")

        # 正值判断
        present_fitted[present_fitted < 0] = 0 if self.is_condition else present_fitted[present_fitted < 0]
        # 模型评价
        print("   Fit model %s >>>>> CC: %s | NSE: %s | RSR: %s | PBIAS: %s" % (
            self.model_info,
            np.corrcoef(self.ref_dataset.values.reshape(-1), present_fitted)[0, 1],
            NSE(self.ref_dataset.values.reshape(-1), present_fitted),
            get_rsr(self.ref_dataset.values.reshape(-1), present_fitted),
            PBIAS(self.ref_dataset.values.reshape(-1), present_fitted)
        ))

        return present_fitted
