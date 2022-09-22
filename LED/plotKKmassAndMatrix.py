import matplotlib.pyplot as plt
import KKmodes as modes
import numpy as np
from graphing import plotting
from vanilla import oscillatorBase

#---------------
R = 0.4
nPoints = 100
nNodes = 20

dm21_2 = 7.42 * 10 ** (-5)
dm31_2 = 2.51 * 10 ** (-3)
#%%---------------
Dms = np.logspace(-3, 0, nPoints)
ms0 = np.zeros((nPoints,nNodes))
ms1 = np.zeros((nPoints,nNodes))
ms2 = np.zeros((nPoints,nNodes))

V0 = np.zeros((nPoints,nNodes))
V1 = np.zeros((nPoints,nNodes))
V2 = np.zeros((nPoints,nNodes))
#%%
x = modes.KKtower(nNodes, Dms[0], R, inverted=False, approx=True)
ms0[0] = x.masses0
ms1[0] = x.masses1
ms2[0] = x.masses2
V0[0] = x.V[0, :]
V1[0] = x.V[1, :]
V2[0] = x.V[2, :]

for i in range(1, nPoints):
    x.update(nNodes, Dms[i], R)
    ms0[i] = x.masses0
    ms1[i] = x.masses1
    ms2[i] = x.masses2
    V0[i] = x.V[0, :]
    V1[i] = x.V[1, :]
    V2[i] = x.V[2, :]

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400, figsize=(6, 6))
colors = [plotting.parula_map_r(i) for i in np.linspace(0, 1, 14)]
#plt.cm.gist_rainbow
#ax.set_prop_cycle(cycler('color', colors))
#cycle = cycler('color', colors)

for i in range(nNodes):
    plotting.niceLinPlot(ax, Dms, ms2[:, i], linestyle='-', color='g')
    plotting.niceLinPlot(ax, Dms, ms1[:, i], linestyle='-', color='r')
    plotting.niceLinPlot(ax, Dms, ms0[:, i], linestyle='-', color='pink')


plt.show()

y = oscillatorBase.Oscillator(10, 10)
print(y.PMNS)
print(y.getOsc(0,0))
