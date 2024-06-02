# coding=utf-8
import numpy as np
import pandas as pd
import glob, os
import re
from RasterHandler import rasterClassify
import matplotlib.pyplot as plt


def classify_sign(x):
    return -2 if x < -2.58 else -1 if -2.58 <= x < -1.96 else 0 if -1.96 <= x <= 1.96 else 1 if 1.96 < x < 2.58 else 2 if x > 2.58 else x


def statistical_trend(npz_list, classify_path, classify_refer_path, classify_miss_value, out_path):
    color = ["#006400", "#32CD32", "#FFFF00", "#FFA500", "#FF0000"]
    writer = pd.ExcelWriter(out_path + "Trend statistical.xlsx")
    for npz_file in npz_list:
        var_name = re.findall(r'[^\\/:*?"<>|\r\n]+$', npz_file)[0][:-4]
        dict = np.load(npz_file)
        mk = dict["mk"]

        # 对全球变化趋势分类
        global_mk = mk.flatten()
        global_mk = np.array(list(map(classify_sign, global_mk)))

        ex_inc_num = np.sum(global_mk == 2)
        sign_inc_num = np.sum(global_mk == 1)
        no_sign_num = np.sum(global_mk == 0)
        sign_dec_num = np.sum(global_mk == -1)
        ex_dec_num = np.sum(global_mk == -2)
        total = np.sum(global_mk == global_mk)

        percentage = [ex_inc_num / total, sign_inc_num / total, no_sign_num / total, sign_dec_num / total,
                      ex_dec_num / total]
        statistical_df = pd.DataFrame(
            {"Trend": ["Extremely significant increase", "Significant increase", "Non-significant",
                       "Significant decrease", "Extremely significant decrease"], "global": percentage})

        out_file = out_path + "plot\\" + var_name + "_global_pie.pdf"
        make_pie(percentage, statistical_df["Trend"], out_file, title="Global %s" % var_name, colors=color,
                 make_legend=True)

        # 获取气候区分类
        y, x = np.shape(mk)
        classify = rasterClassify.get_classify_ndarray(classify_path, x, y)
        classify_num = np.unique(classify)
        classify_num = classify_num[classify_num != classify_miss_value]

        classify_refer = pd.read_csv(classify_refer_path, index_col=0, header=0)

        for i in classify_num:
            print("\rstatistical trend of %s in climate zone %d" % (var_name, i), end="")
            refer = classify_refer.loc[i, "classify"]
            classify_mk = mk.copy()
            classify_mk[classify != i] = np.nan

            classify_mk = classify_mk.flatten()
            classify_mk = np.array(list(map(classify_sign, classify_mk)))

            ex_inc_num_i = np.sum(classify_mk == 2)
            sign_inc_num_i = np.sum(classify_mk == 1)
            no_sign_num_i = np.sum(classify_mk == 0)
            sign_dec_num_i = np.sum(classify_mk == -1)
            ex_dec_num_i = np.sum(classify_mk == -2)
            total_i = np.sum(classify_mk == classify_mk)
            percentage_i = [ex_inc_num_i / total_i, sign_inc_num_i / total_i, no_sign_num_i / total_i,
                            sign_dec_num_i / total_i,
                            ex_dec_num_i / total_i]

            statistical_df[refer] = percentage_i
            out_filei = out_path + "plot\\" + var_name + "_" + refer + "_pie.jpg"
            make_pie(percentage_i, statistical_df["Trend"], out_filei, title="%s %s" % (refer, var_name), colors=color,
                     make_legend=True)
        statistical_df.to_excel(writer, sheet_name=var_name, index=False, float_format="%.4f")
    writer.close()


def make_pie(data, labels, out_path, title=None, colors=None, make_legend=False):
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体样式
    plt.rcParams['font.size'] = '10'  # 设置字体样式
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(6, 6))  # 将画布设定为正方形，则绘制的饼图是正圆

    if colors is not None:
        if len(colors) != len(data):
            colors = None

    explode = [0.01] * len(data)  # 设定各项距离圆心n个半径
    plt.pie(data, colors=colors, explode=explode, autopct='%1.2f%%', pctdistance=1.12)  # 绘制饼图
    plt.axis('equal')  # 设置饼图长宽相等
    if title is not None:
        plt.title(title,y=1.2)
    if make_legend:
        plt.legend(loc='lower center', ncol=len(data) // 2, frameon=False, shadow=False,
                   labels=labels, bbox_to_anchor=(0.5, -0.15))
    plt.savefig(out_path, dpi=300)  # 保存图片
    # plt.show()
    plt.close()


if __name__ == "__main__":
    npz_path = r"..\Backup\GLDAS\interpretation" + "\\"
    npz_list = glob.glob(npz_path + "*.npz")

    out_path = r"..\Backup\GLDAS\interpretation\statistics" + "\\"
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    classify_path = r"..\classify_60_5.tif"
    classify_refer_path = r"..\classifycode_5.csv"

    statistical_trend(npz_list, classify_path, classify_refer_path, 127, out_path)
