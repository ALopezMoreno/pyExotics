import sys

sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
from graphing import plotting
import matplotlib.pyplot as plt
from tqdm import tqdm

nPoints = 10**3
energies = np.logspace(-1.5, 2, nPoints)

d_solar = 245
d_vac = 0

prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 1, 10 - 2, False, False, 0,
                                              ngens=3,
                                              neOverNa=True)
ne_profile = [[2.40008004e+02, 2.30327083e+02, 2.16532919e+02, 1.99417145e+02
                  , 1.83654282e+02, 1.69137389e+02, 1.55767979e+02, 1.43455350e+02
                  , 1.32115968e+02, 1.21672905e+02, 1.12055310e+02, 9.90353602e+01
                  , 8.39977036e+01, 7.12433841e+01, 6.04256969e+01, 5.12505813e+01
                  , 4.00326636e+01, 2.87984254e+01, 2.07168154e+01, 1.26402126e+01
                  , 6.54128226e+00, 2.43514965e+00, 9.03791401e-02],
              [2720.078125, 2720.078125, 5440.15625, 5440.15625, 5440.15625
                  , 5440.15625, 5440.15625, 5440.15625, 5440.15625, 5440.15625
                  , 5440.15625, 10880.3125, 10880.3125, 10880.3125, 10880.3125
                  , 10880.3125, 21760.625, 21760.625, 21760.625, 43521.25
                  , 43521.25, 87042.5, 348170.]]

amp = np.ones(nPoints, dtype=complex)

for k, e in tqdm(enumerate(energies), total=len(energies)):

    P_base = np.zeros((3,3), dtype=complex)
    for i in range(3):
        P_base[i, i] = 1
    for j in range(len(ne_profile[0])):

        prop.update_hamiltonian(e * 10 ** -3, ne_profile[0][j], ngens=3, neOverNa=True)
        #U = prop.vMixingMatrix  # .conjugate().T
        U_m = prop.mixingMatrix  # .conjugate().T
        P = np.zeros((3,3), dtype=complex)
        for i in range(prop.generations):
            for h in range(prop.generations):
                for g in range(prop.generations):
                    phase = prop.eigenvals[g] * ne_profile[1][j] * 2 / (2*prop.E)
                    P[i, h] += U_m[i, g].conjugate() * U_m[h, g] * np.exp(-phase * 2j)
        np.matmul(P, P_base, out=P_base)

    amp[k] = P_base[0, 0]

probs = np.abs(amp)**2

# energy smearing:
window_size = len(energies) // 10
if (window_size % 2) == 0:
    window_size += 1
weights = np.ones(window_size) / window_size
pad_size = (window_size - 1) // 2
probs_padded = np.pad(probs, (pad_size, pad_size), mode='edge')
probs_avg = np.convolve(probs_padded, weights, mode='valid')
for i in range(pad_size):
    probs_avg[i] = np.mean(probs[:i + pad_size + 1])
    probs_avg[-i - 1] = np.mean(probs[-i - pad_size - 1:])


# LMA approx
th12 = np.arcsin(np.sqrt(0.308))
th13 = np.arcsin(np.sqrt(0.022))
lma_condition = np.cos(2 * th12)
msw_condition = 1
beta = (2 * np.sqrt(2) * 5.3948e-5 * np.cos(th13) ** 2 * 245 * energies * 10 ** -3) / (7.42 * 10 ** (-5))
matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
probsLMA = np.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12))  # - 2*np.cos(th13)**2

fig, ax = plt.subplots(dpi=250)
plotting.niceLinPlot(ax, energies, np.abs(amp) ** 2, logx=True, logy=False, color='red', linestyle='', marker='o', alpha=0.4,
                     markersize=1)
plotting.niceLinPlot(ax, energies, probsLMA, logy=False, color='gold', linewidth=1.7, label=r'$P_{LMA}$')
plotting.niceLinPlot(ax, energies, probs_avg, logy=False, color='lightsteelblue', linewidth=1.5, label=r'$P_{avg}$')

plt.show()

data = np.column_stack((energies, probs))
np.savetxt('../oscillationProbs/testMatterEff.txt', data, delimiter="\t", fmt='%.9f')