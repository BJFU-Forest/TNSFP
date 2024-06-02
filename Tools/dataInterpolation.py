import numpy as np
import pandas as pd


def season_transformer(df, date_list_name, type=0):
    """
    返回季节尺度结果
    :param df: 日/月尺度数据 (pandas.DataFrame)
    :param date_list_name: 日期列名称 (str)
    :param type:  数据平均类型 (int) 0: 求平均, 1: 求和
    :return:
    """
    dates = df[date_list_name].values
    month_list = [pd.to_datetime(date).month - 1 for date in dates]
    month_refer = np.array(
        ["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"])
    month_list = [month_refer[m] for m in month_list]
    df[date_list_name] = month_list
    if type == 0:
        df = df.groupby(date_list_name).mean()
    elif type == 1:
        df = df.groupby(date_list_name).mean()
        df = df * 3
    else:
        raise Exception("请选择正确的季节平均算法 (0: 求平均, 1: 求和)")
    df = df.reindex(["MAM", "JJA", "SON", "DJF"])
    return df


def annual_transformer(df, date_list_name, type=0):
    """
    返回年尺度结果
    :param df: 日/月尺度数据 (pandas.DataFrame)
    :param date_list_name: 日期列名称 (str)
    :param type:  数据平均类型 (int) 0: 求平均, 1: 求和
    :return:
    """

    def to_year_index(df):
        years = [pd.to_datetime(date).year for date in df._stat_axis.values.tolist()]
        df["Year"] = years
        df = df.set_index("Year")
        return df

    if type == 0:
        df = df.set_index(date_list_name).resample("Y").mean()
    elif type == 1:
        df = df.set_index(date_list_name).resample("Y").sum()
    else:
        raise Exception("请选择正确的年平均算法 (0: 求平均, 1: 求和)")
    df = to_year_index(df)
    return df


def fill_nan_data(df, date_name, data_name):
    """
    用相同月份的均值填充缺失值
    :param df: 数据表(pd.df) 必须项[date, data]
    :param date_name: 日期列名称(str)
    :param data_name: 数据列名称(str)
    :return:
    """
    n = 0
    for d in np.isnan(df[data_name]):
        if d:
            df.loc[n, data_name] = np.nanmean(
                df[data_name][df[date_name].dt.month == df.loc[n, date_name].month].values)
        n += 1
    return df


def drop_data_less_than_year(df, date_name):
    """
    抛弃不足一年的数据
    :param df: 数据表(pd.df) 必须项[date, data]
    :param date_name: 日期列名称(str)
    :return:
    """
    start_line = 0
    for date in df[date_name]:
        if date.month == 1:
            break
        start_line += 1

    end_line = 0
    n = 0
    for date in df[date_name]:
        n += 1
        if date.month == 12:
            end_line = n
    df = df.iloc[start_line:end_line]
    return df