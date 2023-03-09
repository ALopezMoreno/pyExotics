import sys
sys.path.append('../')

from tqdm import tqdm
import numpy as np
from HamiltonianSolver import customPropagator
from matplotlib import pyplot as plt
from graphing import plotting
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def matterHamiltonian(density, ngens):
    #  nominal matter hamiltonian
    H = np.zeros((ngens, ngens))
    H[0, 0] = density * 1.3948e-5 * np.sqrt(2)
    if ngens>3:
        for i in range(3, ngens):
            H[i, i] = -2/3*H[0, 0]
    return H

def extractMixingAngles(mixMatrix):
    #  get mixing angles from generic unitary 3x3 matrix

    mMatrix = np.abs(mixMatrix)
    th13 = np.arcsin(mMatrix[0, 2])

    if np.cos(th13) != 0:
        th12 = np.arccos(mMatrix[0, 0] / np.cos(th13))
        th23 = np.arccos(mMatrix[2, 2] / np.cos(th13))
    else:
        if mMatrix[1, 1] != 0:
            th12 = np.arctan(mMatrix[1, 0] / mMatrix[1, 1])
            th23 = np.arcos(mMatrix[1, 1] / np.cos(th12))
        else:
            th12 = np.arctan(mMatrix[2, 0] / mMatrix[2, 1])
            th23 = np.arcos(mMatrix[1, 0] / np.sin(th12))

    if th13 != 0 and th23 != 0 and th12 != 0 and np.cos(th12) != 0 and np.cos(th23) != 0:
        mod = mMatrix[1, 1]**2
        numerator = np.cos(th12)**2 * np.cos(th23)**2 + np.sin(th12)**2 * np.sin(th23)**2 * np.sin(th13) ** 2 - mod
        denominator = 2*np.cos(th12)*np.sin(th12)*np.cos(th23)*np.sin(th23)*np.sin(th13)
        dcp_abs = np.arccos(numerator / denominator)
        dcp = - dcp_abs * np.sign(mixMatrix[0, 2].imag)
    else:
        dcp = 0

    return np.array([np.sin(th12)**2, np.sin(th23)**2, np.sin(th13)**2, dcp])

n = 2.6
l = 295
E = 0.65
npoints = 300

matterH = matterHamiltonian(n, 3)

prop = customPropagator.HamiltonianPropagator(matterH, l, E)
dcps = np.linspace(-1*np.pi+0.001, 1*np.pi-0.001, npoints)
#dcps = np.linspace(-1, 1, npoints)
vals = np.zeros((2, npoints))
shift = np.zeros(npoints)

for i, delta in tqdm(enumerate(dcps)):
    prop.IH = False
    prop.antinu = False
    prop.mixingPars[3] = delta
    prop.update()
    #mixPars_NH = extractMixingAngles(prop.mixingMatrix)
    propNu = prop.getOsc(1, 0)
    prop.antinu = True
    prop.update()
    propANu = prop.getOsc(1, 0)
    vals[0, i] = propNu - propANu

    prop.IH = True
    prop.antinu = False
    prop.update()
    #mixPars_IH = extractMixingAngles(prop.mixingMatrix)
    propNu = prop.getOsc(1, 0)
    prop.antinu = True
    prop.update()
    propANu = prop.getOsc(1, 0)
    vals[1, i] = propNu - propANu

for i, delta in tqdm(enumerate(vals[0, :])):
    differences = np.absolute(vals[1, :] - delta)
    index = differences.argmin()

    #print(differences[index] / delta)
    if differences[index] < 0.0001:
        shift[i] = dcps[index] - dcps[i]
        if shift[i] > 1*np.pi:
            shift[i] -= 2*np.pi*1
        elif shift[i] < -1*np.pi:
            shift[i] += 2*np.pi*1
    else:
        shift[i] = np.nan
#print('NH parameters are:  ' + str(vals[0, 5]))
#print('IH parameters are:  ' + str(vals[1, 5]))

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)

plotting.niceLinPlot(ax, dcps, vals[0, :], logx=False, logy=False, color='red', label='NH')
plotting.niceLinPlot(ax, dcps, vals[1, :], logx=False, logy=False, color='blue', label='IH')
plotting.niceLinPlot(ax, dcps, np.abs(vals[1, :] - vals[0, :]), logx=False, logy=False, color='black', label=r'diff', linestyle='--')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction="inout")
ax.tick_params(which='both', top=True, right=True)

ax.set_xlim(-1*np.pi, 1*np.pi)
plt.title(r'Difference in probability between $\nu$ and $\bar{\nu}$ channels (HK)')
plt.legend(loc='upper left')
plt.xlabel(r'$\delta_{CP}$')
plt.ylabel(r'$\Delta P_{\mu \rightarrow e}$')
plt.savefig("../images/probDiff.png")

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)

ax.set_xlim(-1*np.pi, 1*np.pi)
ax.set_ylim(-1*np.pi, 1*np.pi)
plt.ylabel(r'Shift (NH$\rightarrow$IH)')
plt.xlabel(r'$\delta_{CP}(NH)$')
plt.title(r'Required shift to find degenerate IH point (HK)')

plt.axvline(x=-1.602, color='goldenrod', linewidth=1.5, label='Asimov A')
plt.axvline(x=0, color='lightseagreen', linewidth=1.5, label='Asimov B')

mindeltasA = np.absolute(dcps + 1.602)
mindeltasB = np.absolute(dcps)
argA = shift[np.where(mindeltasA < 0.01)]
Ax = np.ones(len(argA))
argB = shift[np.where(mindeltasB < 0.01)]
Bx = np.zeros(len(argB))

plotting.niceLinPlot(ax, dcps, shift, logx=False, logy=False, color='r',
                    linestyle="", marker='o', markersize=1)

plt.plot(-1.602*Ax, argA, linestyle="", marker='o', color='goldenrod')
plt.plot(Bx, argB, linestyle="", marker='o', color='lightseagreen')
ax.tick_params(which='both', direction="inout")
ax.tick_params(which='both', top=True, right=True)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.legend()
#plt.savefig("../images/HK_sindcp_hierarchy_mass_effect_shift_NH.png")

for i, delta in tqdm(enumerate(vals[1, :])):
    differences = np.absolute(vals[0, :] - delta)
    index = differences.argmin()

    #print(differences[index] / delta)
    if differences[index] < 0.0001:
        shift[i] = dcps[index] - dcps[i]
        if shift[i] > 1*np.pi:
            shift[i] -= 2*np.pi*1
        elif shift[i] < -1*np.pi:
            shift[i] += 2*np.pi*1
    else:
        shift[i] = np.nan

#fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)

ax.set_xlim(-1*np.pi, 1*np.pi)
ax.set_ylim(-1*np.pi, 1*np.pi)
#plt.ylabel(r'Shift (IH$\rightarrow$NH)')
plt.ylabel(r'Shift')
plt.xlabel(r'$\delta_{CP}(IH)$')
plt.title(r'Required shift to find degenerate NH point (HK)')

plt.axvline(x=-1.602, color='goldenrod', linewidth=1.5, label='Asimov A')
plt.axvline(x=0, color='lightseagreen', linewidth=1.5, label='Asimov B')

mindeltasA = np.absolute(dcps + 1.602)
mindeltasB = np.absolute(dcps)
argA = shift[np.where(mindeltasA < 0.01)]
print(argA)
Ax = np.ones(len(argA))
argB = shift[np.where(mindeltasB < 0.01)]
Bx = np.zeros(len(argB))


ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plotting.niceLinPlot(ax, dcps, shift, logx=False, logy=False, color='b',
                     linestyle="", marker='o', markersize=1)
ax.tick_params(which='both', direction="inout")
ax.tick_params(which='both', top=True, right=True)
plt.plot(-1.602*Ax, argA, linestyle="", marker='o', color='goldenrod')
plt.plot(Bx, argB, linestyle="", marker='o', color='lightseagreen')

plt.legend()
plt.savefig("../images/HK_dcp_hierarchy_mass_effect_shift_both.png")


"""
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)
plotting.niceLinPlot(ax, dcps, th12s[0, :], logx=False, logy=False, color='red', label='NH')
plotting.niceLinPlot(ax, dcps, th12s[1, :], logx=False, logy=False, color='blue', label='IH')
plotting.niceLinPlot(ax, dcps, th12s[1, :] - th12s[0, :], logx=False, logy=False, color='black', label=r'diff='+str(th12s[1, 0]-th12s[0, 0]), linestyle='--')


ax.set_xlim(0, 1)
plt.legend(loc='upper left')
plt.xlabel(r'$\theta_{12}$')
plt.ylabel(r'${\theta_{12}}^{eff}$')
plt.savefig("../images/th12_hierarchy_mass_effect.png")
"""