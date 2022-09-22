from vanilla import oscillatorBase
import KKmodes as modes
from graphing import plotting
import numpy as np
import matplotlib.pyplot as plt

E = 0.6
LbL = 295
SbL = 0.280
bins = 200
E_smear = 3 * 0.5
n_smear = 10

vanilla = oscillatorBase.Oscillator(LbL, E, smearing=[E_smear, n_smear], inverted=False)
#get nominal smearing
P_base = vanilla.getOsc(1, 1) # This is the channel we choose

R = np.logspace(-1, 2.5, bins)
m = np.logspace(-3, 1.5, bins)
PL = np.zeros((bins, bins))
PS = np.zeros((bins, bins))


kmode = modes.KKtower(10, 1, 1, inverted=False)
oscS = modes.KKoscillator(SbL, E, kmode, smear=[E_smear, n_smear], inv=False)
oscL = modes.KKoscillator(LbL, E, kmode, smear=[E_smear, n_smear], inv=False)

for i in range(bins):
    for j in range(bins):
        oscS.KKmodes.update(10, m[j], R[i])
        PS[i, j] = oscS.getOsc(1, 1)  # THIS IS ALWAYS MU-MU
        oscL.KKmodes.update(10, m[j], R[i])
        PL[i, j] = oscL.getOsc(1, 1)  # This is the channel we choose
    print("\r", "done with row %i out of %i" % (i, bins))

P = np.abs(np.clip(PL/PS, 0, 1) - P_base)
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400, figsize=(5.5, 4.42))
cont = plotting.plot2Dcontour(ax, m, R, P, logx=True, logy=True)
ax.set_box_aspect(1)
cbar = fig.colorbar(cont)
plt.show()
print('done')
