import KKmodes as modes
import numpy as np
from graphing import plotting
import matplotlib.pyplot as plt
from vanilla import oscillatorBase as base
from nonUnitary import sterileOsc

nPoints = 10000
E = np.ones(nPoints)  # np.logspace(-2, 2, nPoints)
L = np.logspace(0, 4, nPoints)

P = np.zeros(nPoints)
P1 = np.zeros(nPoints)

x = modes.KKtower(5, 0.3, 0.4, inverted=False) #  0.2 in latest plot
propagator = modes.KKoscillator(L[0], E[0], x, smear=[0.5 * 3, 1000], inv=False)
propOriginal = base.Oscillator(L[0], E[0], smearing=[0.5 * 3, 1000], inverted=False)

for i in range(1, nPoints):
    propagator.update(L[i], E[i])
    propOriginal.update(L[i], E[i])
    P[i] = propagator.getOsc(1, 1)  # 0 0 = nu_e survival probability
    P1[i] = propOriginal.getOsc(1, 1)


#print(x.V)

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400, figsize=(5.5, 5))
colors = [plotting.parula_map_r(i) for i in np.linspace(0, 1, 14)]

ax.axhline(0, color='black')
#plotting.niceLinPlot(ax, L, P1, logy=False, marker='.', markersize=1, ls='', color='g')
plotting.niceLinPlot(ax, L, P1, logy=False, marker='.', markersize=1, ls='', color='b', alpha=1)

ax.axvline(295 / 0.6, color='r')
ax.axvline(810 / 1.8, color='g')

ax.set_ylabel("P")
ax.set_xlabel("L/E")
ax.set_box_aspect(1)
plt.show()
