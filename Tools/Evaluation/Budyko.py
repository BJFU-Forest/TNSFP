# coding=utf-8
# coding=utf-8
import numpy as np
import pandas as pd
from sympy import *


def get_n(P, R, ET0):
    n = symbols('n')
    n = nsolve((P - R - ((P * ET0) / (P ** n + ET0 ** n) ** (1 / n))), 1)
    return n


# def get_n(P, R, ET0):
#     n = symbols('n')
#     n = solve((1 + n * ET0 / P) / (1 + n * ET0 / P + (ET0 / P) ** -1) - ((P - R) / P), n)
#     return n

def get_Epsilon(n,P,R,ET0):
    phi = ET0 / P
    ep = (1 + 2 * phi + 3 * n * (phi ** 2)) / (1 + phi + n * (phi ** 2))
    ee = -(phi + 2 * n * (phi ** 2)) / (1 + phi + n * (phi ** 2))
    en = (n * (phi ** 2)) / (1 + phi + n * (phi ** 2))
    return ep, ee, en

def get_contributions(n1,P1,R1,ET01, n2,P2,R2,ET02, ep, ee, en):
    dn = n2-n1
    dp = P2-P1
    dR = R2-R1
    dET0 = ET02-ET01

    cRp = ep*dp*R1/P1/dR
    cRet = ee*dET0*R1/ET01/dR
    cRn = en*dn*R1/n1/dR
    return cRp, cRet, cRn


n = get_n(559.42, 166.55, 935.33)
print(n)
ep, ee, en = get_Epsilon(n, 559.42, 166.55, 935.33)
print(ep, ee, en)
cRp, cRet, cRn = get_contributions(n,576.67, 169.96, 939.45, n, 546.96, 164.28, 931.64, 1.82, -0.82, -1.26)
print(cRp, cRet, cRn)