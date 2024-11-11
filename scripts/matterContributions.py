import numpy as np
from graphing import plotting
from HamiltonianSolver import customPropagator
import matplotlib.pyplot as plt
import matplotlib

npoints = 500
E = 2*10**-2
L = 1

prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L, E, False, False, density=0, neOverNa=True)

densities = np.logspace(-1,7, npoints)
matrices = np.empty((npoints, 3, 3))

for i, dens in enumerate(densities):
    prop.update_hamiltonian(E, density=dens)
    prop.update()
    U = prop.mixingMatrix
    for k in range(3):
        for j in range(3):
            matrices[i, j, k] = np.abs(U[j, k])**2


fig, ax = plt.subplots(3, 1, dpi=200, figsize=(7.5, 6), sharex=True)

for i in range(3):
    eterm = matrices[:,0, i]
    muterm = eterm + matrices[:,1, i]
    tauterm = muterm + matrices[:,2, i]

    plotting.niceLinPlot(ax[i], densities, eterm, logy=False, color='red')
    plotting.niceLinPlot(ax[i], densities, muterm, logy=False, color='royalblue')
    plotting.niceLinPlot(ax[i], densities, tauterm, logy=False, color='lime')

    ax[i].fill_between(densities, np.zeros(npoints), eterm, color='red', alpha=0.65, label=r'$\nu_e$')
    ax[i].fill_between(densities, eterm, muterm, color='royalblue', alpha=0.2, label=r'$\nu_\mu$')
    ax[i].fill_between(densities, muterm, tauterm, color='greenyellow', alpha=0.25, label=r'$\nu_\tau$')

    plotting.makeTicks(ax[i], densities, ydata=[0, 1], allsides=False, xnumber=8, ynumber=5)
    #locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    #ax[i].xaxis.set_minor_locator(locmin)
    #ax[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.subplots_adjust(hspace=0.03)  # You can adjust the value of hspace as needed
for i in range(3):
    ax[i].tick_params(axis='x', which='major', direction='inout', length=8)
    ax[i].set_yticks([ax[i].get_yticks()[1],ax[i].get_yticks()[-2]])

ax[2].set_xlabel(r'$N_e$ $(n_e/N_a)$', fontsize=20)
ax[0].set_ylabel(r'$\nu_1$', fontsize=20)
ax[1].set_ylabel(r'$\nu_2$', fontsize=20)
ax[2].set_ylabel(r'$\nu_3$', fontsize=20)
ax[0].set_title(r'Flavour content of propagation states ($E_\nu=20$MeV)', fontsize=18)

# Add arrow etc
ax[1].axvline(x=500, linewidth=3, color='black', ymax=0.85, ymin=0.55)
arrow_start = (500, 0.85/2+0.55/2)  # Coordinates of the starting point of the arrow
arrow_length = 499     # Length of the arrow
ax[1].axvline(1.3, linestyle='--', linewidth=1.5, color='black')
ax[1].axvline(1.3, linewidth=4, ymax=matrices[1, 0, 1], color='darkred', alpha=1)


ax[1].annotate('', xy=(arrow_start[0] - arrow_length, arrow_start[1]),
            xytext=arrow_start, arrowprops=dict(arrowstyle='->',
                                                linewidth=3,
                                                mutation_scale=20), color='black')

for ax2 in ax:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(1.5)

legend = ax[2].legend(fontsize=18, framealpha=1)
plt.savefig('../images/flavourContributionsWide.pdf', format='pdf', bbox_inches='tight')