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
    H[0, 0] = density * 5.3948e-5 * np.sqrt(2)
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
        mod = mMatrix[1, 2]**2
        numerator = np.sin(th12)**2 * np.cos(th23)**2 + np.sin(th12)**2 * np.sin(th23)**2 * np.sin(th13) ** 2 - mod
        denominator = 2*np.cos(th12)*np.sin(th12)*np.cos(th23)*np.sin(th23)*np.sin(th13)

        dcp_abs = np.arccos(numerator / denominator)
        dcp = - dcp_abs * np.sign(mixMatrix[0, 2].imag)

    else:
        dcp = 0
    return np.array([th12, th23, th13, dcp])


#%%
h4 = matterHamiltonian(0, 4)
h3 = matterHamiltonian(0, 3)
magMix=0.001
E = 10**3
prop = customPropagator.HamiltonianPropagator(h3, 295/0.6*E, E*10**-3)
prop.masses = [0, np.sqrt(7.42 * 10 ** (-5)), np.sqrt(2.51 * 10 ** (-3)), 10**6]
prop.mixingPars = [np.arcsin(np.sqrt(0.307)), np.arcsin(np.sqrt(0.022)), np.arcsin(np.sqrt(0.561)),
                   np.arcsin(np.sqrt(magMix)), np.arcsin(np.sqrt(magMix)), np.arcsin(np.sqrt(magMix)), -1.601, 0.0, 0.0]
prop.generations = 4
prop.new_hamiltonian(h4)
prop.update()
npoints = 200
nthrows = 1

#  calculate matter effect for varying densities
start = -1
end = 5

#energies = [0.2, 0.9, 1.05, 7, 8, 10]
#centralE = 10**-2
#energies = np.random.normal(10**-2, 1*10**-3, nthrows)
#energies = np.linspace(10**-2-1*10**-3, 10**-2+1*10**-3, nthrows)
#m_sterile = [10**3] #  np.linspace(10**3, 10**6, nthrows)
rho = np.logspace(start, end, npoints)
angles = np.zeros((nthrows, npoints, 4))
probL = np.zeros(npoints)
lengths = np.logspace(0, 4, npoints)

for k in tqdm(range(nthrows)):
    #prop.E = energies[k]*10**-3
    #prop.E = E
    # prop.masses[3] = m_sterile[k]
    for i in range(npoints):
        h3 = matterHamiltonian(rho[i], 4)
        if i == 10:
            print(h3)
        prop.new_hamiltonian(h3)
        #angls = extractMixingAngles(prop.mixingMatrix)
        p_osc = [prop.eigenvals[0], prop.eigenvals[1], prop.eigenvals[2], prop.eigenvals[3]*10**-10]

        for j in range(4):
            angles[k][i][j] = p_osc[j]

osc = np.average(angles, axis=0)
#  plotting
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)
plt.grid(True, which="both", axis='x', linestyle='--', linewidth=0.8)

colourses = ['r', 'b', 'gold', 'cyan']
#legends = [r'$\hat{\theta}_{12}$', r'$\hat{\theta}_{23}$', r'$\hat{\theta}_{13}$']
#legends = [r'$P_{ee}$', r'$P_{e\mu}$', r'$P_{\mu\mu}$']
legends = [r'$E_{1m}$', r'$E_{2m}$', r'$E_{3m}$', r'$E_{4m}\times 10^{-10}$']
#ax.axvspan(10**-1, 10**2, color='black', alpha=0.2)
#plotting.draw_line_between_verticals(ax, 10**-2, 10**2, draw_arrow=True, thickness=1.5, xscale='log')


#ax.set_ylim(0, 1)
ax.set_xlim(rho[0],rho[-1])

#  plot formatting stuff
#plt.title(r'Non-U effective mixing angles in matter for pp, Be$^7$, pep and Be$^8$ neutrinos')
plt.title(r'Eigenstates of matter hamiltonian for $E_{\nu}=$ 1GeV $m_s=1$MeV')
plt.suptitle(r'Magnitude of mixing: $sin^2\theta_{i4}=$' + str(magMix))
for i in range(4):
    ax.axhline(y=osc[0, i], color=colourses[i], linestyle='--', linewidth=1)
    plotting.niceLinPlot(ax, rho, osc[:, i], logy=True, color=colourses[i], linewidth=1.5,
                         label=legends[i])

for k in range(nthrows):
    for i in range(4):
        plotting.niceLinPlot(ax, rho, osc[:, i], logy=False, color=colourses[i], linewidth=1, label=None)

ax.set_xlabel(r'$n_{e}$', fontsize=15)
#ax.set_ylabel(r'$\sin^2(2\hat{\theta}_{ij})$', fontsize=15)
ax.set_ylabel(r'$\lambda_{eff}$', fontsize=15)
plt.legend(loc='upper left')

#ax.set_xticks(list(ax.get_xticks()) + [2.6])
#fig.canvas.draw()
#labels = [item.get_text() for item in ax.get_xticklabels()]
#labels[-1] = r'$\bigoplus$'
#ax.set_xticklabels(labels)

plt.savefig('../images/matterEffect_eigenvalues4flavour_smallMixing_MeV.png')
