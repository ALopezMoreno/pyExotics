import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from vanilla import oscillatorBase
import experiments
from graphing import plotting
from nonUnitary import sterileOsc
import os
from vanilla import unitarityTests
import matplotlib.ticker as ticker

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})


frames = 200
parula = plotting.parula_map

x = oscillatorBase.Oscillator(1, 1)
shapes = unitarityTests.UnitarityGeometries(x.PMNS, 0, 0)

dcp = np.linspace(np.pi/2, 2*np.pi, frames)
for i in range(len(dcp)):
    x.dcp = dcp[i]
    x.setPMNS()
    shapes.matrix = x.PMNS
    shapes.get_shapes()

    fig, ax = plt.subplots(nrows=2, ncols=3, dpi=300, sharex=False)


    for k in range(3):

        ax[0, 0].plot(shapes.e_mu[k:k+2].real, shapes.e_mu[k:k+2].imag, '-r', linewidth=0.75)
        ax[0, 1].plot(shapes.e_tau[k:k + 2].real, shapes.e_tau[k:k + 2].imag, '-r', linewidth=0.75)
        ax[0, 2].plot(shapes.mu_tau[k:k + 2].real, shapes.mu_tau[k:k + 2].imag, '-r', linewidth=0.75)


        ax[1, 0].plot(shapes.one_two[k:k+2].real, shapes.one_two[k:k+2].imag, '-r', linewidth=0.75)
        ax[1, 1].plot(shapes.one_three[k:k + 2].real, shapes.one_three[k:k + 2].imag, '-r', linewidth=0.75)
        ax[1, 2].plot(shapes.two_three[k:k + 2].real, shapes.two_three[k:k + 2].imag, '-r', linewidth=0.75)


        for j in range(2):
            ax[j, k].set_xlim(-3.5, 4.5)
            ax[j, k].tick_params(which='both', direction="inout")
            ax[j, k].set_box_aspect(1)
            ax[j, k].xaxis.set_major_locator(ticker.LinearLocator(5))
            ax[j, k].yaxis.set_major_locator(ticker.LinearLocator(5))

    ax[0, 0].set_ylim(-0.5, 0.5)
    ax[0, 1].set_ylim(-0.5, 0.5)
    ax[0, 2].set_ylim(-8, 8)

    ax[1, 0].set_ylim(-0.2, 0.2)
    ax[1, 1].set_ylim(-3, 3)
    ax[1, 2].set_ylim(-8, 8)

    ax[0, 0].set_title(r'$T_{e\mu}$', loc='left')
    ax[0, 1].set_title(r'$T_{e\tau}$', loc='left')
    ax[0, 2].set_title(r'$T_{\mu\tau}$', loc='left')

    ax[1, 0].set_title(r'$T_{12}$', loc='left')
    ax[1, 1].set_title(r'$T_{13}$', loc='left')
    ax[1, 2].set_title(r'$T_{23}$', loc='left')
    plt.tight_layout()

    fig.suptitle(r"$\delta_{CP}=$" + ' ' + str(round(dcp[i], 3)))
    plt.savefig('/home/andres/Desktop/pyExotics/movieTriangle/comparison_' + str(i) + ".png")
    plt.close()
    print('done %i of %i' % (i+1, frames), end='\r')
