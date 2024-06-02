# coding=utf-8
import numpy as np
import pandas as pd
import scipy
import math
import random


class WeatherGenerator(object):
    """
    It can be used in climate change studies as a downscaling tool by perturbing their parameters to account for expected
    changes in precipitation and temperature. First, second and third-order Markov models are provided to generate
    precipitation occurrence, and four distributions (exponential, gamma, skewed normal and mixed exponential) are available
    to produce daily precipitation quantity. Precipitation generating parameters have options to be smoothed using Fourier
    harmonics. Two schemes (unconditional and conditional) are available to simulate Tmax and Tmin. Finally, a spectral
    correction approach is included to correct the well-known underestimation of monthly and inter-annual
    variability associated with weather generators. (reference WeaGETS - a MATLAB software)
        · precipitation occurrence
            Markov models
                First-order
                Second-order
                Third-order
        · precipitation quantity
            exponential
            gamma
            skewed normal
            mixed exponential
        · Precipitation generating parameters smoothed
            Fourier harmonics
        · Wave values simulate (like Tmax)
            unconditional
            conditional
    """

    def __init__(self, daily_observe, monthly_estimate, aggregator, corrector):
        """ Default Downscaling constructor.

        :param daily_observe: The Dataset to use as the reference dataset (observation)
        :type daily_observe: pandas.DataFrame
        :param monthly_estimate: The Dataset need to downscaling (estimate)
        :type monthly_estimate: pandas.DataFrame
        :param aggregator:
        :type aggregator:
        :param corrector:
        :type corrector:
        """
        self.daily_observe = daily_observe
        self.monthly_estimate = monthly_estimate
        self.aggregator = aggregator
        self.corrector = corrector
        # 转换时间格式
        self.daily_observe.index = pd.to_datetime(self.daily_observe.index)
        self.monthly_estimate.index = pd.to_datetime(self.monthly_estimate.index)
        # 筛除闰年2.29
        self.daily_observe = self.daily_observe[
            ~((self.daily_observe.index.month == 2) & (self.daily_observe.index.day == 29))]
        # df数据长度
        daily_size = self.daily_observe.shape[0]
        self.year = int(daily_size / 365)

        # 日尺度数据
        self.P_d = self.daily_observe["PRE"]
        self.Tmax_d = self.daily_observe["MAXTEM"]
        self.Tmin_d = self.daily_observe["MINTEM"]
        self.RH_d = self.daily_observe["RHU"]
        self.SR_d = self.daily_observe["RS"]
        self.WIN_d = self.daily_observe["WIN"]

        # 转换为年-日矩阵
        self.P_d = self.P_d.values.reshape(self.year, 365)
        self.Tmax_d = self.Tmax_d.values.reshape(self.year, 365)
        self.Tmin_d = self.Tmin_d.values.reshape(self.year, 365)
        self.RH_d = self.RH_d.values.reshape(self.year, 365)
        self.SR_d = self.SR_d.values.reshape(self.year, 365)
        self.WIN_d = self.WIN_d.values.reshape(self.year, 365)

    description = "Markov temporal downscaling methods"

    def monthly2daily(self, model_path, precipitation_Threshold=0.1):
        """
        :param precipitation_Threshold: Event (Precipitation) threshold is the amount of precipitation used to determine
                               whether a given day is wet or not (0.1mm is the most commonly used value)
        :type precipitation_Threshold: float
        :param model_path: model parameters backup path
        :type model_path: str [path]
        :return:
        """
        predict_date = self.monthly_estimate.index
        predict_date = pd.date_range(str(predict_date.year[0]), str(predict_date.year[-1] + 1), freq="D", inclusive="left")
        downscaling = pd.DataFrame(columns=self.monthly_estimate.columns.values, index=predict_date)

        PnP, occurrence = self.precipitation_occurrence(precipitation_Threshold)
        dist_param = self.event_distribution(self.P_d)
        states, pre = self.precipitation_generator(occurrence, dist_param, precipitation_Threshold)

        A0, B0, ay0, sy0 = self.temperature_analysis(PnP)
        Tmax, Tmin = self.temperature_generator(states, A0, B0, ay0, sy0)

        ay1, sy1 = self.cycle_value_analysis(self.RH_d, PnP)
        rh = self.cycle_value_generator(states, ay1, sy1, val_name="RHU")
        rh[rh > 0.95] = 0.95
        rh[rh < 0] = 0

        ay2, sy2 = self.cycle_value_analysis(self.SR_d, PnP)
        sr = self.cycle_value_generator(states, ay2, sy2, val_name="RS")
        sr[sr < 0] = 0

        dist_win_w, dist_win_d = self.random_value_analysis(self.WIN_d, PnP)
        win = self.random_value_generator(states, dist_win_w, dist_win_d, val_name="WIN")
        win[win < 0] = 0

        # model parameters backup
        model_backup_file = model_path + "model_parameters.npz"
        np.savez(model_backup_file, occurrence=occurrence, dist_param=dist_param, A0=A0, B0=B0, ay0=ay0, sy0=sy0,
                 ay1=ay1, sy1=sy1, ay2=ay2, sy2=sy2, dist_win_w=dist_win_w, dist_win_d=dist_win_d)

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

    def precipitation_occurrence(self, precipitation_Threshold=0.1):
        """
        to generate event (precipitation) occurrence by Markov chain
        :param precipitation_Threshold: Precipitation threshold is the amount of precipitation used to determine whether a given
                               day is wet or not (0.1mm is the most commonly used value)
        :type precipitation_Threshold: float
        :return:
        """

        def _get_pnp_matrix(data):
            # generate matrix of precip - no precip (PnP size[nyears 365])
            year, day = data.shape
            PnP = np.zeros([year, day])
            PnP[data > precipitation_Threshold] = 1  # a value of 1 means that significant precipitation occured

            data[(data < precipitation_Threshold) & (data > 0)] = 0
            data[data > precipitation_Threshold] = data[data > precipitation_Threshold] - precipitation_Threshold

            PnP[data < 0] = np.nan
            return PnP, data

        def _transition(PnP):
            """
            This function calculates the transition matrix [a] from a time series of daily precipitation.
            The matrix has a dimension a(4,D), where D is the number of days analyzed in a calendar year (usually 365).

            :param PnP: matrix of precip - no precip
            :type PnP: np.array [year, day]
            :return:
            """
            year, day = PnP.shape
            PnPp = np.zeros([year, day])
            PnPp[:, 1:] = PnP[:, :day - 1]
            PnPp[1:, 0] = PnP[:year - 1, day - 1]

            """
            a(0,:) correspond to a00:  transition from a dry day on day n-1 to a dry day on day n
            a(1,:) correspond to a01:  transition from a dry day on day n-1 to a wet day on day n
            a(2,:) correspond to a10:  transition from a wet day on day n-1 to a dry day on day n
            a(3,:) correspond to a11:  transition from a wet day on day n-1 to a wet day on day n
            """

            a01 = np.zeros([year, day])
            a10 = a01.copy()
            a11 = a01.copy()

            # subtract PnP frpm PnPp.  -1 correspond to a01  1 correspond to a10
            difP = PnPp - PnP
            a01[difP == -1] = 1
            a10[difP == 1] = 1

            #  scalar multiplication of PnP and PnPp.  1 correspond to a11
            prodP = np.multiply(PnP, PnPp)
            a11[prodP == 1] = 1

            # transitions are stored in matrix a(4,D)
            a = np.zeros([4, day])
            a[1, :] = np.nansum(a01, axis=0)
            a[2, :] = np.nansum(a10, axis=0)
            a[3, :] = np.nansum(a11, axis=0)

            a[0, :] = year - np.nansum(a[1:], axis=0) - np.isnan(prodP).sum(axis=0)

            return a

        def _markov_probability(a):
            # calculate average p00 and p10 using maximum likelihood estimator
            # on 14-days periods (Woolhiser and Pegram, 1979)
            day = a.shape[1]
            n14 = round(day / 14)
            A00, A01, A10, A11 = [], [], [], []
            for i in range(n14):
                A00.append(a[0, 14 * i: 14 * (i + 1)].sum())
                A01.append(a[1, 14 * i: 14 * (i + 1)].sum())
                A10.append(a[2, 14 * i: 14 * (i + 1)].sum())
                A11.append(a[3, 14 * i: 14 * (i + 1)].sum())
            A00 = np.asarray(A00)
            A01 = np.asarray(A01)
            A10 = np.asarray(A10)
            A11 = np.asarray(A11)

            ap00 = A00 / (A01 + A00)
            ap10 = A10 / (A10 + A11)
            ap01 = 1 - ap00
            ap11 = 1 - ap10

            return ap00, ap01, ap10, ap11

        PnP, self.P_d = _get_pnp_matrix(self.P_d)

        a = _transition(PnP)

        ap00, ap01, ap10, ap11 = _markov_probability(a)

        occurrence = np.asarray([ap00, ap01, ap10, ap11])
        occurrence = occurrence.repeat(14, axis=1)
        occurrence = np.c_[occurrence, occurrence[:, -1]]
        return PnP, occurrence

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

    def temperature_analysis(self, PnP):
        # generate the fourier estimates of average and standard deviations
        def _cov2(x1, x2, lag):
            def _cov(X, Y, k):
                if X.shape != Y.shape:
                    raise Exception("length of X and Y must be same")
                # cov = np.cov(X, Y)[0, 1]  # m = mean(X)
                cov = (X * Y).sum() / (X.shape[0] - ((X == 0) & (Y == 0)).sum() - k - 1)
                return cov

            if lag == 0:
                M = np.ones([2, 2])
                M[0, 1] = _cov(x1, x2, lag)
                M[1, 0] = M[0, 1]

            else:
                M = np.ones([2, 2])
                M[0, 0] = _cov(x1[lag:], x1[:-lag], lag)
                M[0, 1] = _cov(x2[lag:], x1[:-lag], lag)
                M[1, 0] = _cov(x1[lag:], x2[:-lag], lag)
                M[1, 1] = _cov(x2[lag:], x2[:-lag], lag)
            return M

        def _chol(Ma, Mb):
            A = Mb.dot(np.linalg.inv(Ma))
            C = Ma - Mb.dot(np.linalg.inv(Ma).dot(Mb.T))

            """
            the correlation matrix has to be positive definite. When dealing with 
            measured data, observational errors, biases and missing data all 
            contaminate the true correlation and in some cases this results in 
            negative eigenvalues and the cholesky decomposition is not possible.
            thus, if the eigenvalues are not positive, simple set it close to zero.
            """
            D = np.zeros([2, 2])
            (D[0, 0], D[1, 1]), V = np.linalg.eigh(C)
            D[D < 0] = 0.000001
            C = V.dot(D.dot(np.linalg.inv(V)))

            # lambda=diag(eig(C))
            dd = np.zeros([2, 2])
            (dd[0, 0], dd[1, 1]), ee = np.linalg.eigh(C)
            # B=chol(lambda)
            B = ee.dot(np.sqrt(dd).dot(ee.T))
            return A, B

        resX1, X1C, X1D = self.stats(self.Tmax_d, PnP)
        resX2, X2C, X2D = self.stats(self.Tmin_d, PnP)

        aC = np.c_[X1C[0], X2C[0], X1C[1], X2C[1]]
        sC = np.c_[X1C[2], X2C[2], X1C[3], X2C[3]]
        aD = np.c_[X1D[0], X2D[0], X1D[1], X2D[1]]
        sD = np.c_[X1D[2], X2D[2], X1D[3], X2D[3]]

        t = np.arange(365) + 1
        T = 365 / (2 * np.pi)
        ay, sy = [], []
        for i in range(aC.shape[1]):
            ay.append(aC[0, i] + aC[1, i] * np.sin(t / T + aD[0, i]) + aC[2, i] * np.sin(2 * t / T + aD[1, i]))
            sy.append(sC[0, i] + sC[1, i] * np.sin(t / T + sD[0, i]) + sC[2, i] * np.sin(2 * t / T + sD[1, i]))
        ay = np.asarray(ay)
        sy = np.asarray(sy)

        M0 = _cov2(resX1, resX2, lag=0)
        M1 = _cov2(resX1, resX2, lag=1)
        A, B = _chol(M0, M1)
        return A, B, ay, sy

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

    def cycle_value_analysis(self, X, PnP):
        # generate the fourier estimates of average and standard deviations
        resX, C, D = self.stats(X, PnP)

        aC = C[:2].T
        sC = C[2:].T
        aD = D[:2].T
        sD = D[:2].T

        t = np.arange(365) + 1
        T = 365 / (2 * np.pi)
        ay, sy = [], []
        for i in range(aC.shape[1]):
            ay.append(aC[0, i] + aC[1, i] * np.sin(t / T + aD[0, i]) + aC[2, i] * np.sin(2 * t / T + aD[1, i]))
            sy.append(sC[0, i] + sC[1, i] * np.sin(t / T + sD[0, i]) + sC[2, i] * np.sin(2 * t / T + sD[1, i]))
        ay = np.asarray(ay)
        sy = np.asarray(sy)
        return ay, sy

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

    def random_value_analysis(self, X, PnP):
        def find_best_distribution(data):
            """
            快速拟合多种分布并进行双样本KS检验/t检验，返回最合适的分布，及对应的参数。
            :param data: 需要拟合的数据
            :type data: np.array-like
            :return:
            """
            # Turn off code warnings (this is not recommended for routine use)
            import warnings
            warnings.filterwarnings("ignore")

            size = len(data)

            dist_list = [
                'norm',  # 正态分布
                'expon',  # 指数分布
                'gamma',  # 伽马分布
                'lognorm',  # 对数正态分布
                'weibull_min',  # 威布尔分布
                'genextreme',  # 广义极值分布
                'genpareto',  # 广义pareto分布
                'gumbel_r',  # Gumbel分布
                'pearson3',  # 皮尔逊Ⅲ型分布
            ]
            dist_list_used = dist_list.copy()

            # Set up empty lists to store results
            ks_values = []
            p_values = []
            param_list = []

            # Loop through candidate distributions
            for dist_name in dist_list:
                print("\rDistribution: %s" % dist_name, end="")
                try:
                    # Set up distribution and get fitted distribution parameters
                    dist = getattr(scipy.stats, dist_name)
                    param = dist.fit(data)
                    # Generate random numbers
                    r = dist.rvs(*param, size=size)

                    # # calculate KS
                    # eva_value, p_value = scipy.stats.ks_2samp(data, r)
                    # calculate T-test
                    eva_value, p_value = scipy.stats.ttest_ind(data, r)

                    if np.isnan(eva_value):
                        dist_list_used.remove(dist_name)
                        print(
                            "\nDistribution %s cannot be applied to fit the given data! (Error: ks=NAN)" % dist_name)
                    else:
                        ks_values.append(eva_value)
                        p_values.append(1 - p_value)
                        param_list.append(param)
                except Exception as e:
                    dist_list_used.remove(dist_name)
                    print("\nDistribution %s cannot be applied to fit the given data! (Error: %s)" % (dist_name, e))
            print("\r", end="")
            # find the best distribution
            best_dist_ind = np.nanargmin(p_values)
            return dist_list_used[best_dist_ind], p_values[best_dist_ind], param_list[best_dist_ind]

        nP = 1 - PnP
        nP[np.isnan(nP)] = 0

        Xw = X.copy()
        Xw[PnP == 0] = np.nan

        Xd = X.copy()
        Xd[nP == 0] = np.nan

        month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month = np.nancumsum(month)
        dist_param_w = {}
        dist_param_d = {}
        for i in range(len(month) - 1):
            Xwi = Xw[:, month[i]:month[i + 1]].copy()
            Xwi = Xwi[~np.isnan(Xwi)].flatten()
            dist_w, p_value, param_w = find_best_distribution(Xwi)
            dist_param_w[i + 1] = {
                "distribution": dist_w,
                "parameter": param_w
            }

            Xdi = Xd[:, month[i]:month[i + 1]].copy()
            Xdi = Xdi[~np.isnan(Xdi)].flatten()
            dist_d, p_value, param_d = find_best_distribution(Xdi)
            dist_param_d[i + 1] = {
                "distribution": dist_d,
                "parameter": param_d
            }
        return dist_param_w, dist_param_d

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

    def stats(self, X, P):
        """

        :param X:
        :param P:
        :return:
        """
        nP = 1 - P
        nP[np.isnan(nP)] = 0

        Xw = X.copy()
        Xw[P == 0] = np.nan
        aw = np.nanmean(Xw, axis=0)
        sw = np.nanstd(Xw, axis=0, ddof=1)
        aw[np.isnan(aw)] = 0
        sw[np.isnan(sw)] = 0

        Xd = X.copy()
        Xd[nP == 0] = np.nan
        ad = np.nanmean(Xd, axis=0)
        sd = np.nanstd(Xd, axis=0, ddof=1)
        ad[np.isnan(ad)] = 0
        sd[np.isnan(sd)] = 0

        aXw, aCw, aDw = self.fourier(aw)
        aXd, aCd, aDd = self.fourier(ad)
        sXw, sCw, sDw = self.fourier(sw)
        sXd, sCd, sDd = self.fourier(sd)
        C = np.asarray([aCw, aCd, sCw, sCd])
        D = np.asarray([aDw, aDd, sDw, sDd])

        residw = (Xw - aXw) / sXw
        residd = (Xd - aXd) / sXd

        residw[np.isnan(residw)] = 0
        residd[np.isnan(residd)] = 0
        resid = residw + residd
        resid = resid.flatten()

        return resid, C, D

    def fourier(self, aY, level=2):
        n = aY.shape[0]
        t = np.arange(n) + 1
        T = n / (2 * np.pi)

        X = [
            np.ones(n),
        ]
        for i in range(level):
            X.append(np.sin((i + 1) * t / T))
            X.append(np.cos((i + 1) * t / T))

        X = np.asarray(X).T

        invX = np.linalg.pinv(X)
        coefp = invX.dot(aY)

        C0 = coefp[0]
        C, D = [C0], []
        Y = C0
        for i in range(level):
            d = math.atan(coefp[2 * (i + 1)] / coefp[2 * i + 1])
            c = coefp[2 * i + 1] / math.cos(d)
            Y += c * np.sin((i + 1) * t / T + d)

            D.append(d)
            C.append(c)

        C = np.asarray(C)
        D = np.asarray(D)
        return Y, C, D

    def event_distribution(self, daily_matrix):
        def find_best_distribution(data):
            """
            快速拟合多种分布并进行双样本KS检验/t检验，返回最合适的分布，及对应的参数。
            :param data: 需要拟合的数据
            :type data: np.array-like
            :return:
            """
            # Turn off code warnings (this is not recommended for routine use)
            import warnings
            warnings.filterwarnings("ignore")

            size = len(data)

            dist_list = [
                'norm',  # 正态分布
                'expon',  # 指数分布
                'gamma',  # 伽马分布
                'lognorm',  # 对数正态分布
                'weibull_min',  # 威布尔分布
                'genextreme',  # 广义极值分布
                'genpareto',  # 广义pareto分布
                'gumbel_r',  # Gumbel分布
                'pearson3',  # 皮尔逊Ⅲ型分布
            ]
            dist_list_used = dist_list.copy()

            # Set up empty lists to store results
            ks_values = []
            p_values = []
            param_list = []

            # Loop through candidate distributions
            for dist_name in dist_list:
                print("\rDistribution: %s" % dist_name, end="")
                try:
                    # Set up distribution and get fitted distribution parameters
                    dist = getattr(scipy.stats, dist_name)
                    param = dist.fit(data)
                    # Generate random numbers
                    r = dist.rvs(*param, size=size)

                    # # calculate KS
                    # eva_value, p_value = scipy.stats.ks_2samp(data, r)
                    # calculate T-test
                    eva_value, p_value = scipy.stats.ttest_ind(data, r)

                    if np.isnan(eva_value):
                        dist_list_used.remove(dist_name)
                        print(
                            "\nDistribution %s cannot be applied to fit the given data! (Error: ks=NAN)" % dist_name)
                    else:
                        ks_values.append(eva_value)
                        p_values.append(1 - p_value)
                        param_list.append(param)
                except Exception as e:
                    dist_list_used.remove(dist_name)
                    print("\nDistribution %s cannot be applied to fit the given data! (Error: %s)" % (dist_name, e))
            print("\r", end="")
            # find the best distribution
            best_dist_ind = np.nanargmin(p_values)
            return dist_list_used[best_dist_ind], p_values[best_dist_ind], param_list[best_dist_ind]

        month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month = np.nancumsum(month)
        dist_param = {}
        daily_matrix = daily_matrix.copy()
        for i in range(len(month) - 1):
            daily_data = daily_matrix[:, month[i]:month[i + 1]]
            daily_data = daily_data[daily_data > 0].flatten()
            dist, p_value, param = find_best_distribution(daily_data)
            dist_param[i + 1] = {
                "distribution": dist,
                "parameter": param
            }
        return dist_param
