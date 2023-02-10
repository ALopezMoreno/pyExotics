import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../')

from HamiltonianSolver import customPropagator
from graphing import plotting


def matterHamiltonian(density, ngens):
    #  nominal matter hamiltonian
    H = np.zeros((ngens, ngens))
    H[0, 0] = density * 1.663787e-5 * np.sqrt(2)
    if ngens>3:
        for i in range(3, ngens):
            H[i, i] = -2*H[0, 0]
    return H

def extractMixingAngles(mixMatrix):
    #  get mixing angles from generic unitary 3x3 matrix
    #  (ignores complex phases)
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

    return np.array([th12, th23, th13])


#%%
h4 = matterHamiltonian(0, 4)
h3 = matterHamiltonian(0, 3)
E = 10
prop = customPropagator.HamiltonianPropagator(h3, 295/0.6*E, E*10**-3)
prop.masses = [0, np.sqrt(7.42 * 10 ** (-5)), np.sqrt(2.51 * 10 ** (-3)), 10**3]
prop.mixingPars = [np.arcsin(np.sqrt(0.307)), np.arcsin(np.sqrt(0.022)), np.arcsin(np.sqrt(0.561)),
                   np.arcsin(np.sqrt(0.)), np.arcsin(np.sqrt(0.)), np.arcsin(np.sqrt(0.)), -1.601, 0.0, 0.0]
prop.generations = 4
prop.new_hamiltonian(h4)
prop.update()
npoints = 200
nthrows = 100

#  calculate matter effect for varying densities
start = -4
end = 7

#energies = [0.2, 0.9, 1.05, 7, 8, 10]
centralE = 10**-2
#energies = np.random.normal(10**-2, 1*10**-3, nthrows)
energies = np.linspace(10**-2-1*10**-3, 10**-2+1*10**-3, nthrows)
m_sterile = [10**3] #  np.linspace(10**3, 10**6, nthrows)
rho = np.logspace(start, end, npoints)
angles = np.zeros((nthrows, npoints, 3))
probL = np.zeros(npoints)
lengths = np.logspace(0, 4, npoints)

for k in tqdm(range(nthrows)):
    #prop.E = energies[k]*10**-3
    prop.E = energies[k]
    # prop.masses[3] = m_sterile[k]
    for i in range(npoints):
        h3 = matterHamiltonian(rho[i], 4)
        prop.new_hamiltonian(h3)
        #angls = extractMixingAngles(prop.mixingMatrix)
        p_osc = [prop.getOsc(0, 0), prop.getOsc(0, 1), prop.getOsc(1, 1)]

        for j in range(3):
            angles[k][i][j] = p_osc[j]

osc = np.average(angles, axis=0)
#  plotting
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)
plt.grid(True, which="both", axis='x', linestyle='--', linewidth=0.8)

colourses = ['r', 'b', 'gold']
#legends = [r'$\hat{\theta}_{12}$', r'$\hat{\theta}_{23}$', r'$\hat{\theta}_{13}$']
legends = [r'$P_{ee}$', r'$P_{e\mu}$', r'$P_{\mu\mu}$']

ax.axvspan(10**-2, 10**2, color='black', alpha=0.2)
#plotting.draw_line_between_verticals(ax, 10**-2, 10**2, draw_arrow=True, thickness=1.5, xscale='log')


ax.set_ylim(0, 1)
ax.set_xlim(10**(start+1), 10**(end-1))

#  plot formatting stuff
#plt.title(r'Non-U effective mixing angles in matter for pp, Be$^7$, pep and Be$^8$ neutrinos')
plt.title(r'Non-U 10MeV neutrino and $m_s=$ 1keV noMixing(vanilla)')
for i in range(3):
    ax.axhline(y=osc[0, i], color=colourses[i], linestyle='--', linewidth=1.5)
    plotting.niceLinPlot(ax, rho, osc[:, i], logy=False, color=colourses[i], linewidth=0.75,
                         label=legends[i])

for k in range(nthrows):
    for i in range(3):
        plotting.niceLinPlot(ax, rho, osc[:, i], logy=False, color=colourses[i], linewidth=0.75, label=None)

ax.set_xlabel(r'$n_{e}$', fontsize=15)
#ax.set_ylabel(r'$\sin^2(2\hat{\theta}_{ij})$', fontsize=15)
ax.set_ylabel(r'$P_{\alpha \beta}$', fontsize=15)
plt.legend(loc='upper left')

#ax.set_xticks(list(ax.get_xticks()) + [2.6])
#fig.canvas.draw()
#labels = [item.get_text() for item in ax.get_xticklabels()]
#labels[-1] = r'$\bigoplus$'
#ax.set_xticklabels(labels)

plt.savefig('../images/matterEffect_probsAtFirstMinimum_10MeV_ms1KeV_noMixing_smeared.png')
