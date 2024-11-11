import sys
sys.path.append("../")
from HamiltonianSolver import customPropagator
import numpy as np
from tqdm import tqdm
from graphing import plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# FUCTION FOR PLOTTING
def makePlots(probs, title, fig_name, colormap=plotting.parula_map, max=0.1):
    th23_dcp_nu_NH = probs[:, :, int(nbins / 2)]
    th23_dm23_nu_NH = probs[:, int(nbins / 2), :]
    dcp_dm23_nu_NH = probs[int(nbins / 2), :, :]

    common_vmin = np.min(probs)
    common_vmax = np.max(probs)

    if max is None:

        fig, axes = plt.subplots(2, 2, dpi=200, figsize=(8, 8))
        col = axes[0][0].imshow(th23_dcp_nu_NH.T, cmap=colormap, extent=[S23.min(), S23.max(), dcp.min(), dcp.max()],
                                aspect='auto')
        axes[1][0].imshow(th23_dm23_nu_NH.T, cmap=colormap,
                          extent=[S23.min(), S23.max(), dm23.min(), dm23.max()],
                          aspect='auto')
        axes[1][1].imshow(dcp_dm23_nu_NH.T, cmap=colormap, extent=[dcp.min(), dcp.max(), dm23.min(), dm23.max()],
                          aspect='auto')

    else:

        fig, axes = plt.subplots(2, 2, dpi=200, figsize=(8, 8))
        col = axes[0][0].imshow(th23_dcp_nu_NH.T, cmap=colormap, extent=[S23.min(), S23.max(), dcp.min(), dcp.max()],
                          aspect='auto', vmin=common_vmin, vmax=common_vmax)
        axes[1][0].imshow(th23_dm23_nu_NH.T, cmap=colormap,
                          extent=[S23.min(), S23.max(), dm23.min(), dm23.max()],
                          aspect='auto', vmin=common_vmin, vmax=common_vmax)
        axes[1][1].imshow(dcp_dm23_nu_NH.T, cmap=colormap, extent=[dcp.min(), dcp.max(), dm23.min(), dm23.max()],
                          aspect='auto', vmin=common_vmin, vmax=common_vmax)



    axes[0, 1].axis('off')

    if max is not None:
        divider = make_axes_locatable(axes[0, 1])
        cax = divider.append_axes("left", size="5%", pad=0.05)
        colorbar = plt.colorbar(col, cax=cax)



    for i in range(2):
        for j in range(2):
            axes[i][j].set_box_aspect(1)
            if i == j and i == 0:
                axes[i][j].set_ylabel(r'$\delta_{CP}$', fontsize=12)
            elif j < i:
                axes[i][j].set_ylabel(r'$\Delta m_{23}^2$', fontsize=12)
                axes[i][j].set_xlabel(r'$\sin^2\theta_{23}$', fontsize=12)
            else:
                axes[i][j].set_xlabel(r'$\delta_{CP}$', fontsize=12)
    axes[0][0].set_xticklabels([])
    axes[1][1].set_yticklabels([])
    fig.suptitle(title, fontsize=18)

    plt.savefig(fig_name)
    plt.show()

# SET UP SEARCH PARAMETERS:
nbins = 40 # EVEN NUMBER!!!
nens = 20
# SET UP OSCILLATION FIXED PARAMETERS:
L = 295
density = 2.7 / 2
E = 0.6
#energies = np.linspace(0.59, 0.61, nens)
energies = np.asarray([0.6])

S12 = 0.307
S13 = 0.561
dm21 = 7.53 * 10 ** (-5)

# SET UP AXES
S23 = np.linspace(0, 1, nbins)
dcp = np.linspace(-np.pi, np.pi, nbins)
dm23 = np.linspace(2.4 * 10 ** (-3), 2.5 * 10 ** (-3), nbins)


# SET UP PROPAGATOR
prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                              1, False, False, 0, ngens=3)
prop.autoHierarchy = False
prop.update_hamiltonian(E, density, ngens=3, earthCrust=True)

# EMPTY ARRAYS TO CONTAIN PROBABILITIES
P_mu_e_NH = np.empty((nbins, nbins, nbins))
P_mu_mu_NH = np.empty((nbins, nbins, nbins))
P_e_e_NH = np.empty((nbins, nbins, nbins))

P_mubar_ebar_NH = np.empty((nbins, nbins, nbins))
P_mubar_mubar_NH = np.empty((nbins, nbins, nbins))
P_ebar_ebar_NH = np.empty((nbins, nbins, nbins))

P_mu_e_IH = np.empty((nbins, nbins, nbins))
P_mu_mu_IH = np.empty((nbins, nbins, nbins))
P_e_e_IH = np.empty((nbins, nbins, nbins))

P_mubar_ebar_IH = np.empty((nbins, nbins, nbins))
P_mubar_mubar_IH = np.empty((nbins, nbins, nbins))
P_ebar_ebar_IH = np.empty((nbins, nbins, nbins))

diffs = np.empty((nbins, nbins, nbins))
pos = np.empty((nbins,nbins,nbins,3))

# RUN LOOPS
for i, sin2th23 in tqdm(enumerate(S23)):
    for j, delta in enumerate(dcp):
        for k, mass in enumerate(dm23):

            #Normal Hierarchy
            prop.masses = [0, dm21, dm21+mass]
            prop.mixingPars = [np.arcsin(np.sqrt(S12)),
                               np.arcsin(np.sqrt(sin2th23)),
                               np.arcsin(np.sqrt(S13)),
                               delta]

            prop.antinu = False
            prop.update()
            prop.E = energies
            P_mu_e_NH[i, j, k] = np.mean(prop.getOsc(1, 0))
            P_e_e_NH[i, j, k] = np.mean(prop.getOsc(0, 0))
            P_mu_mu_NH[i, j, k] = np.mean(prop.getOsc(1, 1))

            prop.E = E
            prop.antinu = True
            prop.update()
            prop.E = energies
            P_mubar_ebar_NH[i, j, k] = np.mean(prop.getOsc(1, 0))
            P_ebar_ebar_NH[i, j, k] = np.mean(prop.getOsc(0, 0))
            P_mubar_mubar_NH[i, j, k] = np.mean(prop.getOsc(1, 1))
            prop.E = E

            #Inverse Hierarchy
            prop.antinu = False
            prop.masses = [0, dm21, dm21 - mass]

            prop.update()
            prop.E = energies
            P_mu_e_IH[i, j, k] = np.mean(prop.getOsc(1, 0))
            P_e_e_IH[i, j, k] = np.mean(prop.getOsc(0, 0))
            P_mu_mu_IH[i, j, k] = np.mean(prop.getOsc(1, 1))
            prop.E = E

            prop.antinu = True
            prop.update()
            prop.E = energies
            P_mubar_ebar_IH[i, j, k] = np.mean(prop.getOsc(1, 0))
            P_ebar_ebar_IH[i, j, k] = np.mean(prop.getOsc(0, 0))
            P_mubar_mubar_IH[i, j, k] = np.mean(prop.getOsc(1, 1))
            prop.E = E

# Aggregate probs:
P_e_NH = (P_e_e_NH + P_mu_e_NH + P_ebar_ebar_NH + P_mubar_ebar_NH) / 4
P_mu_NH = (P_mu_mu_NH + P_mu_e_NH + P_mubar_mubar_NH + P_mubar_ebar_NH) / 4
P_e_IH = (P_e_e_IH + P_mu_e_IH + P_ebar_ebar_IH + P_mubar_ebar_IH) / 4
P_mu_IH = (P_mu_mu_IH + P_mu_e_IH + P_mubar_mubar_IH + P_mubar_ebar_IH) / 4

P_e_NH_nu = (P_e_e_NH + P_mu_e_NH) / 2
P_e_NH_nubar = (P_ebar_ebar_NH + P_mubar_ebar_NH) / 2
P_mu_NH_nu = (P_mu_mu_NH + P_mu_e_NH) / 2
P_mu_NH_nubar = (P_mubar_mubar_NH + P_mubar_ebar_NH) / 2

P_e_IH_nu = (P_e_e_IH + P_mu_e_IH) / 2
P_e_IH_nubar = (P_ebar_ebar_IH + P_mubar_ebar_IH) / 2
P_mu_IH_nu = (P_mu_mu_IH + P_mu_e_IH) / 2
P_mu_IH_nubar = (P_mubar_mubar_IH + P_mubar_ebar_IH) / 2



# CALCULATE DIFFERENCES
for i in tqdm(range(nbins)):
    for j in range(nbins):
        for k in range(nbins):

            # The far detector loos at either mu_nu or mu_e without distinguishing
            # particle from antiparticle
            # so we want things where the total p->e and total p->mu are equal
            if i!=20 or j !=20 or k!= 20:
                #val_NH = total_NH[i, j, k]
                val_mu_NH_nu = P_mu_NH_nu[i, j, k]
                val_e_NH_nu = P_e_NH_nu[i, j, k]
                val_mu_NH_nubar = P_mu_NH_nubar[i, j, k]
                val_e_NH_nubar = P_e_NH_nubar[i, j, k]

                # these are indices of the smallest difference in the e-like channel
                abs_diff = np.abs(P_e_IH_nu - val_e_NH_nu) + np.abs(P_mu_IH_nu - val_mu_NH_nu) + np.abs(P_e_IH_nubar - val_e_NH_nubar) + np.abs(P_mu_IH_nubar - val_mu_NH_nubar)
                min_index = np.argmin(abs_diff)
                x, y, z = np.unravel_index(min_index, P_e_IH.shape)

                #diff_e = P_e_IH[x, y, z] - val_e_NH
                #diff_mu = P_mu_IH[x, y, z] - val_mu_NH
                diff_total = abs_diff[x, y, z]
                # fetch sum of probabilities
                #diff_e_e_NH = np.abs(P_e_e_IH[alpha, beta, gamma] - val_e_e_NH)
                #diff_mu_mu_NH = np.abs(P_mu_mu_IH[alpha, beta, gamma] - val_mu_mu_NH)
                #diff_mu_e_NH = np.abs(P_mu_e_IH[alpha, beta, gamma] - val_mu_e_NH)

                #diff_ebar_ebar_NH = np.abs(P_ebar_ebar_IH[alpha, beta, gamma] - val_ebar_ebar_NH)
                #diff_mubar_mubar_NH = np.abs(P_mubar_mubar_IH[alpha, beta, gamma] - val_mubar_mubar_NH)
                #diff_mubar_ebar_NH = np.abs(P_mubar_ebar_IH[alpha, beta, gamma] - val_mubar_ebar_NH)

                diffs[i, j, k] = diff_total #diff_e + diff_mu #diff_mu_e_NH + diff_mubar_ebar_NH# diff_e_e_NH + diff_mu_mu_NH + diff_mu_e_NH + diff_ebar_ebar_NH + diff_mubar_mubar_NH
            else:
                val_mu_NH_nu = P_mu_NH_nu[i, j, k]
                val_e_NH_nu = P_e_NH_nu[i, j, k]
                val_mu_NH_nubar = P_mu_NH_nubar[i, j, k]
                val_e_NH_nubar = P_e_NH_nubar[i, j, k]

                # these are indices of the smallest difference in the e-like channel
                abs_diff = np.abs(P_e_IH_nu - val_e_NH_nu) + np.abs(P_mu_IH_nu - val_mu_NH_nu) + np.abs(
                    P_e_IH_nubar - val_e_NH_nubar) + np.abs(P_mu_IH_nubar - val_mu_NH_nubar)
                diffs = abs_diff
"""
np.save('fullRangeMinimumDegeneraciesNuNubar', diffs)
#makePlots(P_mu_e_NH, r'P$_{\mu \rightarrow e}$ (NH)')
#makePlots(P_mubar_ebar_NH, r'P$_{\bar{\mu} \rightarrow \bar{e}}$ (NH)')

#makePlots(P_mu_e_IH, r'P$_{\mu \rightarrow e}$ (IH)')
#makePlots(P_mubar_ebar_IH, r'P$_{\bar{\mu} \rightarrow \bar{e}}$ (IH)')

makePlots(diffs,r'Minimum degeneracy in probabilities between NH and IH - e $\mu$-like', '../images/degeneracies_map_20_bins_constrained_th23.png', colormap='inferno')
#makePlots(P_e_NH, 'P_e_NH', '../images/P_e_NH', max=None)
#makePlots(P_mu_NH, 'P_mu_NH', '../images/P_mu_NH', max=None)
#makePlots(P_e_IH, 'P_e_IH', '../images/P_e_IH', max=None)
#makePlots(P_mu_IH, 'P_mu_IH', '../images/P_mu_IH', max=None)
"""
makePlots(diffs,r'Minimum degeneracy in probabilities between NH and IH - e $\mu$-like', '../images/probdiffsAsimov.png', colormap='inferno')
