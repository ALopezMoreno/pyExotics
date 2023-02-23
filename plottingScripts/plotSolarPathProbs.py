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
    shifts2 = data[:, 2]
    return dcps, shifts1, shifts2

def main():
    #  Usage: python PlotHierarchyChange.py inputFile.txt outputFile.png
    inputFile = sys.argv[1]
    # outputFile = sys.argv[2]

    plotApprox = False
    plotPvsE = True

    energies, probs = load_data(inputFile)

    """    if plotApprox == True:
        x = np.linspace(0, 696340, 200)
        plt.plot(x, solar_density(x), color='blue')
        x = solver.binCentres
        y = solver.binned_ne_profile
        xerr = solver.binWidths/2
    
        plt.errorbar(y=y, x=x, xerr=xerr, fmt='o', color='red', markersize=0.5)
        plt.plot([x[1:]-xerr[1:], x[1:]-xerr[1:]], [y[1:]-np.diff(y), y[1:]], color='red')
        #plt.plot([x[1:]-xerr[1:], x+xerr[1:]], [np.diff(y), np.diff(y)], color='red')
    
        #plt.show()
        # set a QTimer to automatically close the figure after 3 seconds
        timer = QTimer()
        timer.timeout.connect(plt.close)
        timer.start(4000)  # 3000 milliseconds = 3 seconds
    
        # start the main event loop
        plt.get_current_fig_manager().window.show()
        plt.get_current_fig_manager().window.activateWindow()
        plt.get_current_fig_manager().toolbar.setVisible(False)
        plt.get_current_fig_manager().window.setWindowState(
            plt.get_current_fig_manager().window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        plt.get_current_fig_manager().window.raise_()
        plt.rcParams['toolbar'] = 'None'
        plt.show()"""

    if plotPvsE == True:
        angles = np.ones(len(energies)) * 0.308
        th12 = np.arcsin(np.sqrt(angles))
        th13 = np.ones(len(th12)) * np.arcsin(np.sqrt(0.022))
        lma_condition = np.cos(2*th12)
        msw_condition = np.ones(len(th12))

        lma_prob = np.cos(th13)**4 * (1 - 0.5 * np.sin(2*th12)**2)
        msw_prob = np.cos(th13)**4 * np.sin(th12)**2

        beta = 2 * np.sqrt(2) * 1.663787e-5 * np.cos(th13)**2 * 250 * energies * 10**-3 / (7.42 * 10 ** (-5))
        # beta_avg = 2 * np.sqrt(2) * 1.663787e-5 * np.cos(th13)**2 * np.mean(solar_density(np.linspace(0, 696340, 200))) * energies * 10**-3 / ( 7.42 * 10 ** (-5) )
        # print(beta - beta_avg)

        # get averages (window_sizes must be an odd number)
        window_size = len(energies) // 14
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
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
        plt.grid(True, which="both", axis='x', linestyle='--', linewidth=0.8)
        plotting.niceLinPlot(ax, energies, probs, logy=False, color='black', linewidth=1.5, label=r'$P_{ee}$', alpha=0.5)
        plotting.niceLinPlot(ax, energies, probs_avg, logy=False, color='black', linewidth=1.5, label=r'$P_{avg}$')
        plotting.niceLinPlot(ax, energies, lma_prob, logy=False, color='slateblue', linewidth=1.5, label=r'$P_{LMA}$')
        plotting.niceLinPlot(ax, energies, msw_prob, logy=False, color='firebrick', linewidth=1.5, label=r'$P_{MSW}$')
        plotting.niceLinPlot(ax, energies, beta, logy=False, color='goldenrod', linewidth=1.5, label=r'$\beta$')
        # plotting.niceLinPlot(ax, energies, beta, logy=False, color='goldenrod', linewidth=1.5, label=r'$\beta_{avg}$', linestyle ='--')
        ax.axhline(y=np.cos(2 * th12[0]), color='blue', linestyle='--', alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.1)

        ax.set_xlabel(r'$E(MeV)$', fontsize=15)
        # ax.set_ylabel(r'$\sin^2(2\hat{\theta}_{ij})$', fontsize=15)
        ax.set_ylabel(r'$P_{ee}$', fontsize=15)
        plt.title(r'Core-created $\nu_{e}$ survival probability at the solar surface (averaged)', fontsize=12)
        plt.legend(loc='upper left')
        plt.show()

if __name__ == '__main__':
    main()