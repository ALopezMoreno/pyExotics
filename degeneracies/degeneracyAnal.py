import sys
sys.path.append("../")
import numpy as np
from graphing import plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from scipy.ndimage import zoom

# LOAD DATA
diffs = np.load('fullRangeMinimumDegeneracies.npy')

nbins = len(diffs[:, 0, 0])

S23 = np.linspace(0, 1, nbins)
dcp = np.linspace(-np.pi, np.pi, nbins)
dm23 = np.linspace(2.47 * 10 ** (-3), 2.55 * 10 ** (-3), nbins)

nbins = nbins/2

# FUCTION FOR PLOTTING
def makePlots(probs_big, title, fig_name, colormap=plotting.parula_map, max=0.01):
    downsampling_factor = 2
    probs = zoom(probs_big, 1 / downsampling_factor, order=1)

    th23_dcp_nu_NH = probs[:, :, int(nbins / 2)]
    th23_dm23_nu_NH = probs[:, int(nbins / 2), :]
    dcp_dm23_nu_NH = probs[int(nbins / 2), :, :]

    common_vmin = 0
    common_vmax = max

    fig, axes = plt.subplots(2, 2, dpi=200, figsize=(8, 8))
    col = axes[0][0].imshow(th23_dcp_nu_NH.T, cmap=colormap, extent=[S23.min(), S23.max(), dcp.min(), dcp.max()],
                      aspect='auto', vmin=common_vmin, vmax=common_vmax)
    axes[1][0].imshow(th23_dm23_nu_NH.T, cmap=colormap,
                      extent=[S23.min(), S23.max(), dm23.min(), dm23.max()],
                      aspect='auto', vmin=common_vmin, vmax=common_vmax)
    axes[1][1].imshow(dcp_dm23_nu_NH.T, cmap=colormap, extent=[dcp.min(), dcp.max(), dm23.min(), dm23.max()],
                      aspect='auto', vmin=common_vmin, vmax=common_vmax)


    axes[0, 1].axis('off')
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


makePlots(diffs,r'Minimum degeneracy in probabilities between NH and IH - All channels',
          '../images/degeneracies_map_50_bins_constrained_th23_053-063avg.png', colormap='inferno', max=0.05)