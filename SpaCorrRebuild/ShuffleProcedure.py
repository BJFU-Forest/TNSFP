import numpy as np
import pandas as pd
import scipy


def multiSiteCorr(obs, gen, need_fill=False, threshold=0.1, rate=1e3):
    # 填充obs中dry-days
    obs = fillTracePrec(obs, threshold, rate) if need_fill else obs
    # Calculate correlation matrix Cobs for observe data
    Cobs = pd.DataFrame(obs).corr(method="spearman")
    # 基于spearman相关性第一次洗牌
    shuffled_gen = shuffleProcedure(Cobs, gen, need_fill, threshold, rate)
    # 基于pearson相关性调整目标相关性
    shuffled_gen = fillTracePrec(shuffled_gen, threshold, rate) if need_fill else shuffled_gen
    adjust_cos = adjustCorr(obs, shuffled_gen)
    # 基于调整后目标相关性第二次洗牌
    shuffled_gen = shuffleProcedure(adjust_cos, gen, need_fill, threshold, rate)
    return shuffled_gen, adjust_cos


def shuffleProcedure(Cobs, gen, need_fill=False, threshold=0.1, rate=1e3):
    """
    Parameters
    ----------
    Cobs: 目标相关性
    gen： 待调整相关性的矩阵
    need_fill: 是否需要替换0
    threshold: 视为0的阈值，need_fill为True时必须
    rate: 补零时的调整倍率，need_fill为True时必须

    Returns
    -------
    shuffled_gen: 相关性调整后的矩阵
    """
    if need_fill:
        # Fill trace precipitation (<0.1mm) to input data
        X = fillTracePrec(gen, threshold, rate)
        # Rebuilding the spatiotemporal correlation between the generated precipitation series
        shuffled_gen = Iman_Conover(Cobs, X)
        # 还原放大数据
        shuffled_gen[shuffled_gen < threshold * rate] = 0
        shuffled_gen = shuffled_gen / rate
    else:
        shuffled_gen = Iman_Conover(Cobs, gen)
    return shuffled_gen


def fillTracePrec(X, threshold=0.1, rate=1e3):
    """
    Parameters
    ----------
    :param X: input Data - n x k = be a n x k matrix of the generated precipitation, among which k represents the number
     of generated precipitation series from k stations and n represent the total number of days simulated for each month
    :type X: array-like
    :param threshold: threshold of precipitation
    :type threshold: float
    :param rate: rate to enlarge the ranges of trace values to be added to zero precipitation days.
    :type rate: int or flot

    Returns
    -------
    X: input data fill trace precipitation (<0.1mm)
    """
    dry_days = pd.DataFrame(X).where(X < threshold).count(axis=0)
    X = X * rate
    for j, dry_day in enumerate(dry_days):
        X[:, j][X[:, j] < rate * threshold] = np.random.permutation(np.arange(dry_day)) * (
                    rate * threshold - 1) / dry_day
    return X


def Iman_Conover(C, X):  # Li, Z. (2014). doi: 10.1007/s00382-013-1979-2
    """
    Parameters
    ----------
    :parm X: input data
    :type X: array-like
    :param C: desired correlation matrix Cobs (from observe data)
    :type C: array-like

    Returns
    -------
    ranked_x
    """
    # Generate a van der Waerden scores matrix
    S = vanderWaerdenScores(X)
    # Calculate correlation matrix Cobs for generated data
    T = pd.DataFrame(S).corr(method="spearman")
    # 基于spectral decomposition处理非正定矩阵
    C = C if np.all(np.linalg.eigvals(C)) > 0 else modifiedIllMat(C)
    T = T if np.all(np.linalg.eigvals(T)) > 0 else modifiedIllMat(T)
    # Use cholesky decomposition to select lower triangular matrix where LL^(-1)=C
    L = scipy.linalg.cholesky(C, lower=True)
    # Use cholesky decomposition to select lower triangular matrix where QQ^(-1)=T
    Q = scipy.linalg.cholesky(T, lower=True)
    # Calculate R = L * Q^(-1)
    R = L @ np.linalg.inv(Q)
    # Multiply the samples by L to induce the correlations
    # and by R' to remove the effects of unintended correlations in the input sample.
    ranked = S @ R.transpose()
    # Rank the columns of the resulting matrix
    ranks = scipy.stats.rankdata(ranked, method='ordinal', axis=0)
    ranks = ranks.astype(int) - 1
    # Sort the columns of the original sample matrix to match the rank order
    ranked_x = np.zeros(X.shape)
    for j in range(0, X.shape[1]):
        s_temp = np.sort(X[:, j])
        ranked_x[:, j] = s_temp[ranks[:, j]]
    return ranked_x


def vanderWaerdenScores(X):
    """
    Returns
    -------
    S: The ranks of each column in [X] are then calculated to generate a van der Waerden scores matrix (array-like)
    """
    ranks = scipy.stats.rankdata(X, method='ordinal', axis=0)
    n, k = ranks.shape
    S = scipy.stats.norm.ppf(ranks / (n + 1))
    return S


def modifiedIllMat(C):
    C = np.matrix(C)
    e, v = np.linalg.eig(C)
    e[e <= 0] = 1e-10
    Cm = v * np.diag(e) * np.linalg.inv(v)
    Cm = Cm / np.sqrt(np.diag(Cm) * np.diag(Cm).transpose())
    return Cm


def adjustCorr(obs, gen):  # Sun, X. (2020). doi: 10.3390/w12030904
    Cos = pd.DataFrame(obs).corr(method="spearman")
    Cop = pd.DataFrame(obs).corr(method="pearson")
    Ccp = pd.DataFrame(gen).corr(method="pearson")
    param = scipy.stats.linregress(x=Ccp.values.flatten(), y=Cos.values.flatten())
    adjust_cos = (param.slope * Cop.values.flatten() + param.intercept).reshape(Cos.shape)
    return adjust_cos


if __name__ == "__main__":
    # obs = np.array(
    #     [[1.534, 1.534, 1.534, 1.534, 0.489, 0.319], [0.887, 0.489, 0.887, 0.887, 0.157, 0.674],
    #       [0.887, 0, 0.674, 0.887, 0.674, 1.534],
    #      [1.15, 0.319, 0.489, 0.674, 0.157, 1.15], [0.157, 1.534, 0.887, 0.647, 0.319, 0.157],
    #      [0, 0.674, 0.157, 0.157, 1.534, 0.157], [0, 0.887, 0.157, 0.319, 0.674, 0.887],[0, 0.674, 0.489, 1.150, 1.534, 0.489],
    #      [0.319, 0.157, 0.674, 0.319, 0, 0], [0.319, 0.157, 0, 1.150, 1.150, 0.887],
    #      [1.534, 0.887, 1.150, 1.534, 0.489, 1.150], [0.157, 1.150,0, 0.489, 0.319, 0.489],
    #      [0, 0.489, 1.150, 0.489, 0.887, 0], [0.674, 0.319, 0.319, 0, 0.887, 0],
    #      [0.674, 1.150, 1.1534, 0.157, 1.150, 0]]
    # )

    A = range(90)
    A = np.array(A).reshape(15, 6)
    obs = A

    gen = np.array(
        [[1.534, 1.534, 1.534, 1.534, 0.489, 0.319], [0.887, 0.489, 0.887, 0.887, 0.157, 0.674],
         [0, 0.674, 0.489, 1.150, 1.534, 0.489], [0.887, 0, 0.674, 0.319, 0, 0],
         [1.15, 0.319, 0.489, 0.674, 0.157, 1.15], [0.157, 1.534, 0.887, 0.647, 0.319, 0.157],
         [0, 0.674, 0.157, 0.157, 1.534, 0.157], [0, 0.887, 0.157, 0.319, 0.674, 0.887],
         [0.319, 0.157, 0.674, 0.887, 0.674, 1.534], [0.319, 0.157, 0, 1.150, 1.150, 0.887],
         [1.534, 0.887, 1.150, 1.534, 0.489, 1.150], [0.157, 1.150, 1.1534, 0.157, 1.150, 0],
         [0, 0.489, 1.150, 0.489, 0.887, 0], [0.674, 0.319, 0.319, 0, 0.887, 0],
         [0.674, 1.150, 0, 0.489, 0.319, 0.489]]
    )
    adjustX = multiSiteCorr(obs, gen, need_fill=True, threshold=0.1, rate=1e3)
