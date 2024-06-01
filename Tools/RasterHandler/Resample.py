import numpy as np


class Resampling():
    def __init__(self, arr, res, new_res):
        """
        重采样ndarray，修改分辨率
        :param arr: 原始数据 (ndarray)
        :param res: 原始分辨率 (float)
        :param new_res: 采样分辨率 (float)
        :return: new_arr 重采样后ndarray
        """
        self.arr = arr
        self.res = res
        self.new_res = new_res
        if not isinstance(self.arr, np.ndarray):
            self.arr = np.asarray(self.arr)

    def make_new_arr(self):
        rate = self.new_res / self.res
        rows, cols = np.shape(self.arr)
        new_rows = int(rows * rate)
        new_cols = int(cols * rate)
        new_arr = np.full([new_rows, new_cols], np.nan)
        return rate, rows, cols, new_rows, new_cols, new_arr

    def nearest_neighbour(self):
        """最邻近法，不会更改输入单元的值,适用于不适合求平均的离散变量"""
        rate, rows, cols, new_rows, new_cols, new_arr = self.make_new_arr()

        for i in range(new_rows):
            for j in range(new_cols):
                map_y = int((i + 0.5) * rate)
                map_x = int((j + 0.5) * rate)
                if map_y < rows and map_x < cols:
                    new_arr[i, j] = self.arr[map_y, map_x]
        return new_arr

    def bilinear_interpolation(self):
        """双线性插值法:使用四个最邻近输入单元中心的加权平均值来确定输出栅格上的值。适用于连续表面变量"""
        rate, rows, cols, new_rows, new_cols, new_arr = self.make_new_arr()

        for i in range(new_rows):
            for j in range(new_cols):
                y = (i + 0.5) * rate
                x = (j + 0.5) * rate

                map_y_t = int(y)  # 采样点上方坐标
                map_x_l = int(x)  # 采样点左侧坐标
                map_y_b = min(map_y_t+1, rows-1) # 采样点下方坐标
                map_x_r = min(map_x_l+1,cols-1) # 采样点右侧坐标

                weight = (map_y_b - map_y_t) * (map_x_r - map_x_l)

                new_arr[i, j] =(self.arr[map_y_t, map_x_l] * (y - map_y_t) * (x - map_x_l) + self.arr[
                    map_y_t, map_x_r] * (y - map_y_t) * (map_y_b - x) + self.arr[map_y_b, map_x_l] * (
                                            map_y_b - y) * (x - map_x_l) + self.arr[
                                    map_y_b, map_x_r] * (map_y_b - y) * (map_x_r - x)) / weight

    def cubic_convolution_interpolation(self):
        """三次卷积插值法:通过 16 个最邻近输入单元中心及其值来计算加权平均值。倾向于锐化数据的边缘且速度较慢
        网上抄的代码 未验证！！！！！！！！！！
        """
        def BiBubic(x):
            x = abs(x)
            if x <= 1:
                return 1 - 2 * (x ** 2) + (x ** 3)
            elif x < 2:
                return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
            else:
                return 0

        rate, rows, cols, new_rows, new_cols, new_arr = self.make_new_arr()

        for i in range(new_rows):
            for j in range(new_cols):
                y = (i + 0.5) * rate
                x = (j + 0.5) * rate
                map_y = int(y)
                map_x = int(x)
                u = y - map_y
                v = x - map_x
                tmp = 0
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        map_y_ii = min(max(map_y + ii, 0),rows-1)
                        map_x_jj = min(max(map_x + jj, 0),cols-1)
                        tmp += self.arr[map_y_ii, map_x_jj] * BiBubic(ii - u) * BiBubic(jj - v)
                new_arr[i, j] = tmp
        return new_arr
