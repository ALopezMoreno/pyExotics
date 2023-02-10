import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from graphing import plotting

def load_data(filename):
    data = np.loadtxt(filename)
    rho = data[:,0]
    probs = data[:,1:]
    return rho, probs

def main():
    #  Usage: python PlotMatterEffect.py inputFile.txt outputFile.png
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]

    # Get data:
    rho, osc = load_data(inputFile)

    #  Make figure and plot grid
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)
    plt.grid(True, which="both", axis='x', linestyle='--', linewidth=0.8)

    #  Set colours and legends
    colourses = ['r', 'b', 'gold']
    #legends = [r'$\hat{\theta}_{12}$', r'$\hat{\theta}_{23}$', r'$\hat{\theta}_{13}$']
    legends = [r'$P_{ee}$', r'$P_{e\mu}$', r'$P_{\mu\mu}$']

    #  Set axis properties (and allow for solar range arrow)
    ax.axvspan(10**-2, 10**2, color='black', alpha=0.2)
    #plotting.draw_line_between_verticals(ax, 10**-2, 10**2, draw_arrow=True, thickness=1.5, xscale='log')
    ax.set_ylim(0, 1)
    ax.set_xlim((rho[0]), (rho[-1]))

    #  plot 
    #plt.title(r'Non-U effective mixing angles in matter for pp, Be$^7$, pep and Be$^8$ neutrinos')
    plt.title(r'Non-U 10MeV neutrino and 1keV sterile noMixing(vanilla)')
    for i in range(3):
       ax.axhline(y=osc[0, i], color=colourses[i], linestyle='--', linewidth=1.5)
       plotting.niceLinPlot(ax, rho, osc[:, i], logy=False, color=colourses[i], linewidth=1.5, label=legends[i])

    ax.set_xlabel(r'$n_{e}$', fontsize=15)
    #ax.set_ylabel(r'$\sin^2(2\hat{\theta}_{ij})$', fontsize=15)
    ax.set_ylabel(r'$P_{\alpha \beta}$', fontsize=15)
    plt.legend(loc='upper left')

    #  save output
    plt.savefig('../images/' + outputFile)


if __name__ == '__main__':
    main()
