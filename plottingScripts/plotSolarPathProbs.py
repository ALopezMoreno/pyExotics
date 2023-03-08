import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from graphing import plotting

def solar_density(l):
    solar_radius = 696340
    density = 245*np.exp(-10.54*l/solar_radius)
    return density

def vacuum(l):
    return 0.0
def load_data(filename):
    data = np.loadtxt(filename)
    dcps = data[:, 0]
    shifts1 = data[:, 1]

    return np.array([dcps, shifts1])

def main():
    #  Usage: python PlotHierarchyChange.py inputFile.txt outputFile.png
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]

    plotApprox = False
    plotPvsE = True
    arr = load_data(inputFile).transpose()
    filtered_arr = arr[arr[:, 1] >= 0.025]
    energies = filtered_arr[:, 0]
    probs = filtered_arr[:, 1]

    """
    if plotApprox == True:
        x = np.linspace(0, 696340, 200)
        plt.plot(x, solar_density(x), color='blue')
        x = solver.binCentres
        y = solver.binned_ne_profile
        xerr = solver.binWidths/2
    
        plt.errorbar(y=y, x=x, xerr=xerr, fmt='o', color='red', markersize=0.5)
        plt.plot([x[1:]-xerr[1:], x[1:]-xerr[1:]], [y[1:]-np.diff(y), y[1:]], color='red')
        #plt.plot([x[1:]-xerr[1:], x+xerr[1:]], [np.diff(y), np.diff(y)], color='red')
    
        plt.show()
    """

    if plotPvsE == True:
        th12 = np.arcsin(np.sqrt(0.308))
        th13 = np.arcsin(np.sqrt(0.022))
        lma_condition = np.cos(2*th12)
        msw_condition = 1

        lma_prob = np.cos(th13)**4 * (1 - 0.5 * np.sin(2*th12)**2)
        msw_prob = np.sin(th12)**2 #* np.cos(th13)**4

        beta = 2 * np.sqrt(2) * 1.663787e-5 * np.cos(th13)**2 * 180 * energies / (7.42 * 10 ** (-5)) * 10**-3
        #beta = 0.22 * np.cos(th13)**2 * energies * 7/7.42 #* 2.45
        # calculate where beta hits the msw and vacuum average critical values:
        matterAngle = np.cos(2*th12) - beta / np.sqrt((np.cos(2*th12-beta)**2) + np.sin(2*th12)**2)
        probsLMA = np.cos(th13)**4*(0.5+0.5*matterAngle*np.cos(2*th12))

        mindeltasA = np.absolute(beta - lma_condition)
        mindeltasB = np.absolute(beta - msw_condition)
        crit_lma = energies[np.where(mindeltasA < 0.1)]
        crit_msw = energies[np.where(mindeltasB < 0.1)]

        # get averages (window_sizes must be an odd number)
        window_size = len(energies) // 10
        if (window_size % 2) == 0:
            window_size += 1

        print('convolution window size is: ' + str(window_size))

        weights = np.ones(window_size) / window_size
        pad_size = (window_size - 1) // 2
        probs_padded = np.pad(probs, (pad_size, pad_size), mode='edge')
        probs_avg = np.convolve(probs_padded, weights, mode='valid')
        for i in range(pad_size):
            probs_avg[i] = np.mean(probs[:i + pad_size + 1])
            probs_avg[-i - 1] = np.mean(probs[-i - pad_size - 1:])

        #  Make figure and plot grid
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200)
        plt.grid(True, which="both", axis='x', linestyle='--', linewidth=0.8)
        #plotting.niceLinPlot(ax, energies, probs, logy=False, color='slategray', linewidth=1.5, label=r'$P_{ee}$',
        #                     alpha=0.5)
        plotting.niceLinPlot(ax, energies, probs_avg, logy=False, color='lightsteelblue', linewidth=1.5, label=r'$P_{avg}$')
        plotting.niceLinPlot(ax, energies, probs, logy=False, color='black', markersize=1.25, label=r'$P_{ee}$',
                             alpha=0.6, linestyle='', marker='o')
        plotting.niceLinPlot(ax, energies, probsLMA, logy=False, color='gold', linewidth=1.7, label=r'$\beta$')

        if len(crit_lma) and len(crit_msw):
            mean_lma = np.mean(crit_lma)
            mean_msw = np.mean(crit_msw)
            #ax.axvline(x=mean_lma, linestyle='--', color='blue')
            #ax.axvline(x=mean_msw, linestyle='--', color='red')

            plotting.niceLinPlot(ax, energies[energies < mean_lma], lma_prob*np.ones(len(energies[energies < mean_lma])), logy=False, color='blue', linewidth=1.7,
                                 label=r'$P_{LMA}$', alpha=0.8)
            plotting.niceLinPlot(ax, energies[energies > mean_msw], msw_prob*np.ones(len(energies[energies > mean_msw])), logy=False, color='red', linewidth=1.7,
                                 label=r'$P_{MSW}$', alpha=0.8)

            print(mean_lma, mean_msw)
            print(lma_prob, msw_prob)

        ax.set_ylim(0, 1.1)
        #plotting.niceLinPlot(ax, energies, probs_avg, logy=False, color='black', linewidth=1.5, label=r'$P_{avg}$')

        ax.set_xlabel(r'$E(MeV)$', fontsize=15)
        # ax.set_ylabel(r'$\sin^2(2\hat{\theta}_{ij})$', fontsize=15)
        ax.set_ylabel(r'$P_{ee}$', fontsize=15)
        plt.title(r'Core-created $\nu_{e}$ survival probability at the solar surface (averaged)', fontsize=12)
        plt.legend(loc='upper left')
        plt.savefig('../images/' + outputFile)
        plt.show()

if __name__ == '__main__':
    main()
