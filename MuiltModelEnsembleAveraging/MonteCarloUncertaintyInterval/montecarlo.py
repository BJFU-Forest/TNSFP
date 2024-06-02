import numpy as np
import random
from bisect import bisect_right


def random_sampler(x, sigma, condition):
    # random.seed()  # 视情况设置随机种子
    rx = random.normalvariate(x, sigma)
    while condition & (rx < 0).any():
        rx[rx < 0] = random.normalvariate(x[rx < 0], sigma[rx < 0])
    return rx


# 转换为np.ufunc
error_fit = np.frompyfunc(random_sampler, 3, 1)


# Variance of discrete data
def variance(ensemble, weight):
    eX = (ensemble * weight).sum(axis=1)
    deltaX = ensemble - np.expand_dims(eX, axis=1)
    dX = (deltaX ** 2 * weight).sum(axis=1)
    return dX


########################################################################################################################
class MonteCarlo():
    def __init__(self, weight, ensemble, condition=False):
        self.weight = weight
        self.cumulative_prob = np.insert(self.weight.cumsum(), 0, 0)
        self.ensemble = ensemble
        self.std = variance(ensemble, weight) ** 0.5
        self.condition = condition

        # 统一随机数种子
        random.seed(1023)

    def get_interval(self, niter=10000, percent=0.90):
        sampling = np.empty((niter, self.ensemble.shape[0]), dtype=np.array(enumerate).dtype)

        for i in range(niter):
            sampling[i, :] = self.sampler(self.select())

        # ** percent uncertainty interval
        percentile = (1 - percent) / 2
        interval = np.asarray([np.percentile(sampling, percentile * 100, axis=0),
                               np.percentile(sampling, (percent + percentile) * 100, axis=0)]).T
        return interval

    def select(self):
        return min(self.ensemble.shape[1], bisect_right(self.cumulative_prob, random.random())) - 1

    def sampler(self, draw):
        model = self.ensemble[:, draw]
        # std = model.std()
        sampling = random_sampler(model, sigma=self.std, condition=self.condition)
        return sampling
