import pandas as pd
import numpy as np
import joblib
from sklearn import linear_model
from MuiltModelEnsembleAveraging.BayesianModelAveraging import bayesian_model_averaging


class MultiModeEnsembleAveraging(object):
    def __init__(self, observations, gcm_dict, condition_values):
        self.observations = observations
        self.gcm_dict = gcm_dict
        self.condition_values = condition_values
        self.gcm_name = np.asarray(list(self.gcm_dict.keys()))
        self.date = self.gcm_dict[self.gcm_name[0]].index.values
        self.values = self.gcm_dict[self.gcm_name[0]].columns.values

        if not np.array_equal(self.observations.index.values, self.date):
            raise Exception("The Date series of the GCM sets do not match of observation's, please check!!")

    def get_mmea(self, model_backup_path, method="quick_mmea"):
        supported_method = ["quick_mmea", "lmm_mmea", "ridge_mmea", "bma_mmea"]
        if method not in supported_method:
            raise Exception("Method %s is not yet supported" % method)

        # 备份gcm顺序
        np.save(model_backup_path + "gcm_order.npy", self.gcm_name)

        result = pd.DataFrame({"Date": self.date}).set_index("Date")
        for val in self.values:
            # Observations
            obs = self.observations[val].values
            obs = np.expand_dims(obs, axis=1)  # 将一维obs数组转化为二维数组
            # GCM ensemble
            ensemble = np.asarray([self.gcm_dict[gcm][val].values for gcm in self.gcm_name]).T
            if method == "quick_mmea":
                # Independent Weighted Mean by models' error variance (error = x-y)
                weight = self.quick_mmea(obs, ensemble)
                mean = (ensemble * weight).sum(axis=1)
                np.save(model_backup_path + "quick_weight_%s.npy" % val, weight)
            elif method == "lmm_mmea":
                # Lagrange Multiplier Method for weight
                weight = self.lmm_mmea(obs, ensemble)
                mean = (ensemble * weight).sum(axis=1)
                np.save(model_backup_path + "lmm_weight_%s.npy" % val, weight)
            elif method == "ridge_mmea":
                # Ridge Regressions
                model = self.ridge_mmea(obs, ensemble)
                mean = model.predict(ensemble)
                joblib.dump(model, model_backup_path + "ridge_model_%s.pkl" % val)
            elif method == "bma_mmea":
                # Bayesian model averaging use mcmc approximation
                model = self.bma_mmea(obs, ensemble)
                mean = model.predict(ensemble)
                joblib.dump(model, model_backup_path + "bma_model_%s.pkl" % val)
            else:
                raise Exception("Method %s is not yet supported" % method)
            # 正值条件判断
            is_condition = True if val in self.condition_values else False
            mean[mean < 0] = 0 if is_condition else mean[mean < 0]
            result[val] = mean
        return result

    def quick_mmea(self, obs, ensemble):  # Independent Weighted Mean by models' error variance (error = x-y)
        error_var = (ensemble - obs).std(axis=0)
        weight = (1 / error_var) / (1 / error_var).sum()
        return weight

    def lmm_mmea(self, obs, ensemble):  # Lagrange Multiplier Method
        J = self.date.shape[0]
        K = self.gcm_name.shape[0]
        one = np.ones([K, 1])
        A = np.cov((ensemble - obs).T)
        W = np.full([K, 1], 1e-8)
        l = 1e-8  # lambda (Lagrange Multiplier)
        func = 1 / 2 * W.T * A * W - l * (W.T * one - 1)
        weight = 0
        return weight

    def ridge_mmea(self, obs, ensemble):  # Independent Weighted Mean by Ridge Regressions
        Lambdas = np.logspace(-5, 2, 200)
        model = linear_model.RidgeCV(alphas=Lambdas, normalize=True, scoring='neg_mean_squared_error', cv=10)
        model.fit(ensemble, obs)
        return model

    def bma_mmea(self, obs, ensemble):  # Bayesian model averaging
        penalty_par = max(ensemble.shape[0], ensemble.shape[1] ** 2)
        incl_par = 1 / 3
        model = bayesian_model_averaging.LinearMC3(ensemble, obs, penalty_par, incl_par)
        model.select(niter=10000, method="random")
        model.estimate()
        return model


def get_mmea(gcm_dict, condition_values, model_backup_path, method="quick_mmea"):
    supported_method = ["quick_mmea", "lmm_mmea", "ridge_mmea", "bma_mmea"]
    if method not in supported_method:
        raise Exception("Method %s is not yet supported" % method)

    gcm_order = np.load(model_backup_path + "gcm_order.npy", allow_pickle=True)
    date = gcm_dict[list(gcm_dict.keys())[0]].index.values
    values = gcm_dict[list(gcm_dict.keys())[0]].columns.values

    result = pd.DataFrame({"Date": date}).set_index("Date")
    for val in values:
        # GCM ensemble
        ensemble = np.asarray([gcm_dict[gcm][val].values for gcm in gcm_order]).T
        if method == "quick_mmea":
            # Independent Weighted Mean by models' error variance (error = x-y)
            weight = np.load(model_backup_path + "quick_weight_%s.npy" % val, allow_pickle=True)
            mean = (ensemble * weight).sum(axis=1)
        elif method == "lmm_mmea":
            # Lagrange Multiplier Method for weight
            weight = np.load(model_backup_path + "lmm_weight_%s.npy" % val, allow_pickle=True)
            mean = (ensemble * weight).sum(axis=1)
        elif method == "ridge_mmea":
            # Ridge Regressions
            model = joblib.load(model_backup_path + "ridge_model_%s.pkl" % val)
            mean = model.predict(ensemble)
        elif method == "bma_mmea":
            # Bayesian model averaging use mcmc approximation
            model = joblib.load(model_backup_path + "bma_model_%s.pkl" % val)
            mean = model.predict(ensemble)
        else:
            raise Exception("Method %s is not yet supported" % method)
        # 正值条件判断
        is_condition = True if val in condition_values else False
        mean[mean < 0] = 0 if is_condition else mean[mean < 0]
        result[val] = mean
    return result
