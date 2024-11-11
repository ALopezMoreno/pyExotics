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
def LMA_solution(energy, dm21, s_th12, s_th13, N_e):
    th12 = np.arcsin(np.sqrt(s_th12))
    th13 = np.arcsin(np.sqrt(s_th13))
    beta = (2 * np.sqrt(2) * 5.4489e-5 * np.cos(th13) ** 2 * N_e * energy * 10 ** -3) / (dm21)
    matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
    probLMA = np.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) + np.sin(th13) ** 4

    return probLMA

def LMA_solution_4nu(energy, dm21, s_th12, s_th13, a11, N_e=103, N_n=103/2.3):

    th12 = np.arcsin(np.sqrt(s_th12))
    th13 = np.arcsin(np.sqrt(s_th13))
    #th14 = np.arcsin(np.sqrt(s_th14))
    #beta = (2 * np.sqrt(2) * 5.3948e-5 * np.cos(th13) ** 2 * 103 * a11**4 * energy * 10 ** -3) / (dm21)
    beta = (np.sqrt(2) * 5.4489e-5 * energy * 10 ** -3 / dm21) * (a11 ** 2 * 2 * N_e + (1 - a11 ** 2) * N_n) * np.cos(th13)**2
    matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
    probLMA = np.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) * a11**4

    return probLMA


def LMA_solution_HNL(energy, dm21, s_th12, s_th13, a11, a31):
    N_e = 103
    N_n = 103 / 2.3
    th12 = np.arcsin(np.sqrt(s_th12))
    th13 = np.arcsin(np.sqrt(s_th13))

    beta = (np.sqrt(2) * 5.4489e-5 * energy * 10 ** -3 / (dm21)) * (a11**2*2*N_e + (1-a11**2)*N_n) * np.cos(th13)**2
    # beta = np.sqrt(2) * 5.3948e-5 * energy * 10 ** -3 / dm21 * (a11**4 * (2*Ne*np.cos(th13)**2) - 2*a11**2*a31*Ne*np.sin(2*th13))

    matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
    probLMA = (np.cos(th13)**4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) + np.sin(th13)**4) * a11**4

    return probLMA
