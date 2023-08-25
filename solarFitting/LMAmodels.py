import numpy as np
#import theano
#import theano.tensor as T
"""
def LMA_solution(energy, dm21, th12, th13):
    beta = (2 * T.sqrt(2) * 5.3948e-5 * T.cos(th13) ** 2 * 245 / 2 * energy * 10 ** -3) / (7.42 * 10 ** (-5))
    matterAngle = (T.cos(2 * th12) - beta) / T.sqrt((T.cos(2 * th12) - beta) ** 2 + T.sin(2 * th12) ** 2)
    probLMA = T.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * T.cos(2 * th12))

    return probLMA
"""
def LMA_solution(energy, dm21, s_th12, s_th13):
    th12 = np.arcsin(np.sqrt(s_th12))
    th13 = np.arcsin(np.sqrt(s_th13))
    beta = (2 * np.sqrt(2) * 5.3948e-5 * np.cos(th13) ** 2 * 245 / 2 * energy * 10 ** -3) / (dm21)
    matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
    probLMA = np.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) + np.sin(th13) ** 4

    return probLMA

def LMA_solution_4nu(energy, dm21, s_th12, s_th13, s_th14):
    th12 = np.arcsin(np.sqrt(s_th12))
    th13 = np.arcsin(np.sqrt(s_th13))
    th14 = np.arcsin(np.sqrt(s_th14))
    beta = (2 * np.sqrt(2) * 5.3948e-5 * np.cos(th13) ** 2 * 245 / 2 * np.cos(th14)**4 * energy * 10 ** -3) / (7.42 * 10 ** (-5))
    matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
    probLMA = np.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) * np.cos(th14)**4

    return probLMA