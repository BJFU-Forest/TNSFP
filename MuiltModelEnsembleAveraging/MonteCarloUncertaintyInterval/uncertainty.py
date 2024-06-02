import pandas as pd
import numpy as np
import joblib
from ..BayesianModelAveraging import bayesian_model_averaging
from . import montecarlo
from Tools.Tool import create_path, data_standardizing


class HistoricalUncertaintyAnalysisHandler():
    def __init__(self, observations, ensemble_dict, model_backup_path, uncertainty_path, condition_values,
                 normalization=False):
        self.observations = observations
        self.ensemble_dict = ensemble_dict
        self.method_name = np.asarray(list(self.ensemble_dict.keys()))
        self.gcm_name = np.asarray(list(self.ensemble_dict[self.method_name[0]].keys()))
        self.date = self.ensemble_dict[self.method_name[0]][self.gcm_name[0]].index.values
        self.values = self.ensemble_dict[self.method_name[0]][self.gcm_name[0]].columns.values
        self.model_backup_path = model_backup_path
        self.uncertainty_path = uncertainty_path
        self.condition_values = condition_values
        self.normalization = normalization

        if not np.array_equal(self.observations.index.values, self.date):
            raise Exception("The Date series of the GCM sets do not match of observation's, please check!!")

        # 备份method-gcm顺序
        np.save(self.model_backup_path + "ensemble_order.npy", np.asarray([self.method_name, self.gcm_name]))

    def get_unc_result(self, uncertainty_type):
        uncertainty_type_support = ["GCMs", "Downscaling"]
        if uncertainty_type not in uncertainty_type_support:
            raise Exception("Uncertainty_type %s is not yet supported" % uncertainty_type)

        external_iteration, internal_iteration = self.return_iteration_order(uncertainty_type)
        for ext_iter in external_iteration:
            # 模型备份地址
            model_backup_path = self.model_backup_path + uncertainty_type + "_uncertainty/" + ext_iter + "/"
            create_path(model_backup_path)
            # 结果输出地址
            uncertainty_df_path = self.uncertainty_path + uncertainty_type + "_uncertainty/" + ext_iter + "/"
            create_path(uncertainty_df_path)
            # 计算不确定性
            quantify = pd.DataFrame()
            interval = pd.DataFrame({"Date": self.date}).set_index("Date")
            for val in self.values:
                # Observations
                obs = self.observations[val].values

                # GCM ensemble
                if uncertainty_type == "GCMs":
                    ensemble = np.asarray(
                        [self.ensemble_dict[ext_iter][in_iter][val].values for in_iter in internal_iteration]).T
                else:
                    ensemble = np.asarray(
                        [self.ensemble_dict[in_iter][ext_iter][val].values for in_iter in internal_iteration]).T

                # 非负判断
                condition = val in self.condition_values

                # normalization
                if self.normalization:
                    obs, ensemble, eigenvalue = data_normalization(obs, ensemble)
                    # 备份放缩系数
                    np.save(model_backup_path + "normalization_eigenvalue_%s.npy" % val, np.asarray(eigenvalue))

                # Bayesian model averaging use mcmc approximation
                model = bma_mmea(obs, ensemble)
                joblib.dump(model, model_backup_path + "bma_model_%s.pkl" % val)  # backup BMA model

                # MuiltModelEnsembleAveraging Analysis
                val_quantify, val_interval = get_unc_df(model, ensemble, val, internal_iteration, condition=condition)
                quantify = pd.concat([quantify, val_quantify], axis=0)
                interval[[val + "_low", val + "_high"]] = val_interval
            quantify.to_csv(uncertainty_df_path + "quantify.csv", index=False)
            interval.to_csv(uncertainty_df_path + "interval.csv")

    def return_iteration_order(self, uncertainty_type):
        external_iteration = self.method_name if uncertainty_type == "GCMs" else self.gcm_name
        internal_iteration = self.gcm_name if uncertainty_type == "GCMs" else self.method_name
        return external_iteration, internal_iteration


########################################################################################################################
class FutureUncertaintyAnalysisHandler():
    def __init__(self, ensemble_dict, model_backup_path, uncertainty_path, condition_values, normalization=False):
        self.ensemble_dict = ensemble_dict
        self.model_backup_path = model_backup_path
        self.method_name, self.gcm_name = np.load(self.model_backup_path + "ensemble_order.npy",
                                                  allow_pickle=True)  # 读取method-gcm顺序
        self.date = self.ensemble_dict[self.method_name[0]][self.gcm_name[0]].index.values
        self.values = self.ensemble_dict[self.method_name[0]][self.gcm_name[0]].columns.values
        self.uncertainty_path = uncertainty_path
        self.condition_values = condition_values
        self.normalization = normalization

    def get_unc_result(self, uncertainty_type):
        uncertainty_type_support = ["GCMs", "Downscaling"]
        if uncertainty_type not in uncertainty_type_support:
            raise Exception("Uncertainty_type %s is not yet supported" % uncertainty_type)

        external_iteration, internal_iteration = self.return_iteration_order(uncertainty_type)
        for ext_iter in external_iteration:
            # 模型备份地址
            model_backup_path = self.model_backup_path + uncertainty_type + "_uncertainty/" + ext_iter + "/"
            # 结果输出地址
            uncertainty_df_path = self.uncertainty_path + uncertainty_type + "_uncertainty/" + ext_iter + "/"
            create_path(uncertainty_df_path)
            # 计算不确定性
            quantify = pd.DataFrame()
            interval = pd.DataFrame({"Date": self.date}).set_index("Date")
            for val in self.values:
                # GCM ensemble
                if uncertainty_type == "GCMs":
                    ensemble = np.asarray(
                        [self.ensemble_dict[ext_iter][in_iter][val].values for in_iter in internal_iteration]).T
                else:
                    ensemble = np.asarray(
                        [self.ensemble_dict[in_iter][ext_iter][val].values for in_iter in internal_iteration]).T

                # 非负判断
                condition = val in self.condition_values

                # normalization
                if self.normalization:
                    # 读取放缩系数
                    eigenvalue = np.load(model_backup_path + "normalization_eigenvalue_%s.npy" % val, allow_pickle=True)
                    ensemble = np.asarray([data_standardizing(data, 1, eigenvalue=eigenvalue) for data in ensemble])

                # Bayesian model averaging use mcmc approximation
                model = joblib.load(model_backup_path + "bma_model_%s.pkl" % val)

                # MuiltModelEnsembleAveraging Analysis
                val_quantify, val_interval = get_unc_df(model, ensemble, val, internal_iteration, condition=condition)
                quantify = pd.concat([quantify, val_quantify], axis=0)
                interval[[val + "_low", val + "_high"]] = val_interval
            quantify.to_csv(uncertainty_df_path + "quantify.csv", index=False)
            interval.to_csv(uncertainty_df_path + "interval.csv")

    def return_iteration_order(self, uncertainty_type):
        external_iteration = self.method_name if uncertainty_type == "GCMs" else self.gcm_name
        internal_iteration = self.gcm_name if uncertainty_type == "GCMs" else self.method_name
        return external_iteration, internal_iteration


########################################################################################################################
def bma_mmea(obs, ensemble):  # Bayesian model averaging
    penalty_par = max(ensemble.shape[0], ensemble.shape[1] ** 2)
    incl_par = 1 / ensemble.shape[1]
    model = bayesian_model_averaging.LinearMC3(ensemble, obs, penalty_par, incl_par)
    model.select(niter=10000, method="random")
    model.estimate()
    return model




def data_normalization(obs, ensemble):
    max_data = max(obs.max(), ensemble.max())
    min_data = min(obs.min(), ensemble.min())
    eigenvalue = [max_data, min_data]
    obs = data_standardizing(obs, 1, eigenvalue=eigenvalue)
    ensemble = np.asarray([data_standardizing(data, 1, eigenvalue=eigenvalue) for data in ensemble])
    return obs, ensemble, eigenvalue


def get_unc_df(model, ensemble, val, internal_iteration, condition=False):
    # MuiltModelEnsembleAveraging Analysis
    obj = UncertaintyAnalysis(model=model, ensemble=ensemble)
    inner_uncertainty = obj.inner_uncertainty()
    inter_uncertainty = obj.inter_uncertainty()
    bma_uncertainty = obj.bma_uncertainty()
    interval = obj.uncertainty_interval(condition=condition)
    # write table
    temp = np.full(internal_iteration.shape[0], np.nan).tolist()
    val_array = temp.copy()
    val_array[0] = val
    bma_array = temp.copy()
    bma_array[0] = bma_uncertainty
    quantify_df = pd.DataFrame({
        "Val": val_array,
        "Method": internal_iteration,
        "Inner": inner_uncertainty,
        "Inter": inter_uncertainty,
        "BMA": bma_uncertainty,
    })
    return quantify_df, interval


########################################################################################################################
class UncertaintyAnalysis():
    def __init__(self, model, ensemble):
        """

        :param model: LinearMC3 model
        :type model:
        :param ensemble:
        :type ensemble: np.array
        """
        self.model = model
        self.ensemble = ensemble

        self.weight = self._get_weight()
        self.exp = self.ensemble.mean(axis=0)
        self.var = self.ensemble.var(ddof=1, axis=0)

    def inner_uncertainty(self):
        return self.weight * self.var

    def inter_uncertainty(self):
        exp = np.dot(self.weight, self.exp)
        return self.weight * (self.exp - exp) ** 2

    def bma_uncertainty(self):
        return self.inner_uncertainty().sum() + self.inter_uncertainty().sum()

    def uncertainty_interval(self, condition=False):
        obj = montecarlo.MonteCarlo(weight=self.weight, ensemble=self.ensemble, condition=condition)
        interval = obj.get_interval()
        return interval

    def _get_weight(self):
        posterior = self.model.posterior
        weight = np.zeros(self.ensemble.shape[1])
        for key, w in posterior.items():
            weight += np.asarray(key) * w
        weight = weight / weight.sum()
        return weight
