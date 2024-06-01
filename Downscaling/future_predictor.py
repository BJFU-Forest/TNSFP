# coding=utf-8
import numpy as np
import pandas as pd
import glob
import scipy
import joblib


def predict_future_monthly(method, predictors, model_backup_path, is_condition, scaler=None, labels=None):
    method = method.upper()
    method_list = ["MLR", "MLP"]
    if method not in method_list:
        raise Exception("%s method not support!" % method)
    # 读取预测模型类型（是否分解）
    ref_state = np.load(model_backup_path + "ref_state.npy", allow_pickle=True)
    # 读取拟合参数列表
    X_labels = np.load(model_backup_path + "X_label.npy", allow_pickle=True)
    if (scaler is None) or (labels is None):
        # 加载预测因子(X)缩放器
        X_scaler = joblib.load(model_backup_path + "X_min_max_scaler.pkl")
        # 最小最大规范化对原始数据进行线性变换，变换到[0,1]区间
        X_norm = X_scaler.transform(predictors[X_labels].values)
    else:
        new_df = pd.DataFrame(columns=labels, index=predictors.index,
                              data=np.full([predictors.shape[0], labels.shape[0]], np.nan))
        new_df[X_labels] = predictors[X_labels]
        X_norm = scaler.transform(new_df.values)[:, np.in1d(labels, X_labels)]

    # 模型输出
    future_fitted = []

    if method == "MLR":
        # 读取多元线性回归参数
        predictors_dict = np.load(model_backup_path + "mlr_param.npy", allow_pickle=True).item()
        for i in range(len(list(predictors_dict.keys()))):
            # 预测因子筛选
            index_sel = np.in1d(X_labels, predictors_dict[i])
            X_sel = X_norm[:, index_sel]  # 预测因子集
            # 读取模型
            model = joblib.load(model_backup_path + "mlr_imf%d.pkl" % (i + 1))
            # 模型输出
            future_fitted.append(model.predict(X_sel))

    elif method == "MLP":
        if ref_state:
            for i, path in enumerate(glob.glob(model_backup_path + "mlp_y_imf*.pkl")):
                factor_sel = np.load(model_backup_path + "X_imf%s_labels.npy" % (i + 1), allow_pickle=True)
                index_sel = np.in1d(X_labels, factor_sel)
                X_sel = X_norm[:, index_sel]
                # 读取模型
                model = joblib.load(path)
                # 模型输出
                future_fitted.append(model.predict(X_sel))
        else:
            factor_sel = np.load(model_backup_path + "X_original_labels.npy", allow_pickle=True)
            index_sel = np.in1d(X_labels, factor_sel)
            X_sel = X_norm[:, index_sel]
            # 读取模型
            model = joblib.load(model_backup_path + "mlp_y_original.pkl")
            # 模型输出
            future_fitted.append(model.predict(X_sel))

    # 整合IMF预测结果
    y_scaler = joblib.load(model_backup_path + "y_min_max_scaler.pkl")
    future_fitted = y_scaler.inverse_transform(np.asarray(future_fitted).T).T
    future_fitted = np.asarray(future_fitted).sum(axis=0)

    # 正值判断
    future_fitted[future_fitted < 0] = 0 if is_condition else future_fitted[future_fitted < 0]

    return future_fitted


class PredictFutureDaily(object):
    def __init__(self, monthly_estimate, aggregator, corrector):
        self.monthly_estimate = monthly_estimate
        self.aggregator = aggregator
        self.corrector = corrector
        # 转换时间格式
        self.monthly_estimate.index = pd.to_datetime(self.monthly_estimate.index)
        # df数据长度
        self.year = int(self.monthly_estimate.shape[0] / 12)

    description = "Markov temporal downscaling methods"

    def predict(self, model_path, precipitation_Threshold=0.1):
        predict_date = self.monthly_estimate.index
        predict_date = pd.date_range(str(predict_date.year[0]), str(predict_date.year[-1] + 1), freq="D", inclusive="left")
        downscaling = pd.DataFrame(columns=self.monthly_estimate.columns.values, index=predict_date)

        model = np.load(model_path + "model_parameters.npz", allow_pickle=True)
        occurrence = model["occurrence"]
        dist_param = model["dist_param"].item()
        A0 = model["A0"]
        B0 = model["B0"]
        ay0 = model["ay0"]
        sy0 = model["sy0"]
        ay1 = model["ay1"]
        sy1 = model["sy1"]
        ay2 = model["ay2"]
        sy2 = model["sy2"]
        dist_win_w = model["dist_win_w"].item()
        dist_win_d = model["dist_win_d"].item()

        states, pre = self.precipitation_generator(occurrence, dist_param, precipitation_Threshold)

        Tmax, Tmin = self.temperature_generator(states, A0, B0, ay0, sy0)

        rh = self.cycle_value_generator(states, ay1, sy1, val_name="RHU")
        rh[rh > 0.95] = 0.95
        rh[rh < 0] = 0

        sr = self.cycle_value_generator(states, ay2, sy2, val_name="RS")
        sr[sr < 0] = 0

        win = self.random_value_generator(states, dist_win_w, dist_win_d, val_name="WIN")
        win[win < 0] = 0

        # Reconstruction of time series
        predict_date_365 = predict_date[~((predict_date.month == 2) & (predict_date.day == 29))]
        downscaling_365 = pd.DataFrame(data=np.asarray([pre, rh, sr, Tmin, Tmax, win]).T,
                                       columns=self.monthly_estimate.columns.values,
                                       index=predict_date_365)

        downscaling.loc[downscaling_365.index] = downscaling_365.loc[downscaling_365.index]
        downscaling = downscaling.astype(float)
        # 填充在缺失值的行2.29
        downscaling = downscaling.fillna(method="ffill")
        downscaling.index.rename("Date", inplace=True)
        return downscaling

    def precipitation_generator(self, occurrence, dist_param, precipitation_Threshold=0.1):
        states = np.zeros(self.year * 365)
        amounts = states.copy()

        state = 0

        month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month = np.nancumsum(month)

        # go through possible states and run Markov Chain model for each
        for i in range(len(states)):
            state = np.random.choice([0, 1], replace=True, p=occurrence[state * 2: (state + 1) * 2, i % 365])
            states[i] = state
            if state:
                m = np.digitize(i % 365, month)
                dist = getattr(scipy.stats, dist_param[m]["distribution"])
                while amounts[i] <= 0:
                    amounts[i] = dist.rvs(*dist_param[m]["parameter"])
        amounts = self.monthly_correct(amounts, self.monthly_estimate["PRE"].values, self.aggregator["PRE"],
                                       self.corrector["PRE"])
        amounts[amounts < precipitation_Threshold] = 0
        return states, amounts

    def temperature_generator(self, states, A, B, ay, sy):
        ay = np.tile(ay, self.year)
        sy = np.tile(sy, self.year)
        cay = states * ay[:2, :] + (1 - states) * ay[2:, :]
        csy = states * sy[:2, :] + (1 - states) * sy[2:, :]

        cay[0] = self.monthly_correct(cay[0], self.monthly_estimate["MAXTEM"].values, self.aggregator["MAXTEM"],
                                      self.corrector["MAXTEM"]).T
        cay[1] = self.monthly_correct(cay[1], self.monthly_estimate["MINTEM"].values, self.aggregator["MINTEM"],
                                      self.corrector["MINTEM"]).T

        res = np.zeros(2)
        ksi = np.zeros([2, len(states)])
        for i in range(len(states)):
            eps = scipy.randn(2)
            res = A.dot(res) + B.dot(eps)
            ksi[:, i] = res

        """
        the Tmax and Tmin are generated conditioned on each other (Jie Chen modified)
        the smaller standard deviation of Tmax or Tmin is used as a base, and the
        other parameter is generated conditioned on the chosen parameter. If the
        standard deviation of Tmax is larger than or equal to the standard
        deviation of Tmin, daily temperatures are generated by: (case 1)
        Tmin=Mean(min)+Std(min)*rand
        Tmax=Tmin+(Mean(max)-Mean(min))+(Std(max)^2-Std(min)^2)^0.5*rand

        If the standard deviation of Tmax is less than those of Tmin, daily
        temperatures are genareted by: (case 2)
        Tmax=Mean(max)+Std(max)*rand
        Tmin=Tmax-(Mean(max)-Mean(min))-(Std(min)^2-Std(max)^2)^0.5*rand
        """
        Tmax = np.full(len(states), np.nan)
        Tmin = Tmax.copy()

        # case 1
        case1 = csy[0, :] > csy[1, :]
        Tmin[case1] = cay[1, case1] + csy[1, case1] * ksi[1, case1]
        Tmax[case1] = Tmin[case1] + cay[0, case1] - cay[1, case1] + (
                (csy[0, case1] ** 2 - csy[1, case1] ** 2) ** 0.5) * ksi[0, case1]
        # case 2
        Tmax[~case1] = cay[0, ~case1] + csy[0, ~case1] * ksi[0, ~case1]
        Tmin[~case1] = Tmax[~case1] + cay[1, ~case1] - cay[0, ~case1] - (
                (csy[1, ~case1] ** 2 - csy[0, ~case1] ** 2) ** 0.5) * ksi[1, ~case1]
        # range control the Tmin, insure that Tmin is always small than Tmax
        Tmin[Tmax <= Tmin] = Tmax[Tmax <= Tmin] - np.abs(Tmax[Tmax <= Tmin]) * 0.2
        return Tmax, Tmin

    def cycle_value_generator(self, states, ay, sy, val_name):
        ay = np.tile(ay, self.year)
        sy = np.tile(sy, self.year)
        cay = states * ay[0, :] + (1 - states) * ay[1, :]
        csy = states * sy[0, :] + (1 - states) * sy[1, :]

        cay = self.monthly_correct(cay, self.monthly_estimate[val_name].values, self.aggregator[val_name],
                                   self.corrector[val_name])

        ksi = scipy.randn(len(states))

        Y = cay + csy * ksi

        return Y

    def random_value_generator(self, states, dist_param_w, dist_param_d, val_name, condition=True):
        Y = np.zeros(states.shape)

        month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month = np.nancumsum(month)

        # go through possible states and run Markov Chain model for each
        for i in range(len(states)):
            m = np.digitize(i % 365, month)
            if states[i]:
                dist = getattr(scipy.stats, dist_param_w[m]["distribution"])
                Y[i] = dist.rvs(*dist_param_w[m]["parameter"])
                while condition & (Y[i] <= 0):
                    Y[i] = dist.rvs(*dist_param_w[m]["parameter"])

            else:
                dist = getattr(scipy.stats, dist_param_d[m]["distribution"])
                Y[i] = dist.rvs(*dist_param_d[m]["parameter"])
                while condition & (Y[i] <= 0):
                    Y[i] = dist.rvs(*dist_param_d[m]["parameter"])

        Y = self.monthly_correct(Y, self.monthly_estimate[val_name].values, self.aggregator[val_name],
                                 self.corrector[val_name])
        return Y

    def monthly_correct(self, daily, monthly_correct, aggregator, corrector):
        """

        :param daily:
        :param monthly_correct:
        :param aggregator:
        :param corrector: "translation" - 平移 + alpha； "scaling" - 放缩 * alpha
        :return:
        """
        # 时间列表
        predict_date = self.monthly_estimate.index
        predict_date = pd.date_range(str(predict_date.year[0]), str(predict_date.year[-1] + 1), freq="D", inclusive="left")
        predict_date = predict_date[~((predict_date.month == 2) & (predict_date.day == 29))]

        # 校正系数
        daily = pd.DataFrame({"Date": predict_date, "Forecast": daily}).set_index("Date")
        monthly = daily.groupby([daily.index.year, daily.index.month]).agg(aggregator)
        if corrector == "translation":
            alpha = monthly_correct - monthly.values.T
            alpha[np.isnan(alpha)] = 0
            monthly["alpha"] = alpha.T

            # 日值校正
            daily = daily.groupby([daily.index.year, daily.index.month]).apply(
                lambda x: x + monthly.loc[(x.index[0].year, x.index[0].month), "alpha"])

        elif corrector == "scaling":
            alpha = monthly_correct / monthly.values.T
            alpha[np.isnan(alpha)] = 0
            alpha[np.isinf(alpha)] = 0
            monthly["alpha"] = alpha.T

            # 日值校正
            daily = daily.groupby([daily.index.year, daily.index.month]).apply(
                lambda x: x * monthly.loc[(x.index[0].year, x.index[0].month), "alpha"])

        else:
            raise Exception("argument 'corrector' only support 'translation' and 'scaling'")

        return daily["Forecast"].values
