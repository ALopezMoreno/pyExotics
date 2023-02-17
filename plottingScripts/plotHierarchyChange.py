import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from graphing import plotting
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def load_data(filename):
    data = np.loadtxt(filename)
    dcps = data[:, 0]
    shifts1 = data[:, 1]
    shifts2 = data[:, 2]
    return dcps, shifts1, shifts2

def main():
    #  Usage: python PlotHierarchyChange.py inputFile.txt outputFile.png
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]


    dcps, shift1, shift2 = load_data(inputFile)

    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)

    ax.set_xlim(-1*np.pi, 1*np.pi)
    ax.set_ylim(-1*np.pi, 1*np.pi)
    plt.ylabel(r'Shift')
    plt.xlabel(r'$\delta_{CP}$')
    plt.title(r'Required shift to find degenerate IH point (HK)')

    plt.axvline(x=-1.602, color='goldenrod', linewidth=1.5, label='Asimov A')
    plt.axvline(x=0, color='lightseagreen', linewidth=1.5, label='Asimov B')

    mindeltasA = np.absolute(dcps + 1.602)
    mindeltasB = np.absolute(dcps)
    argA = shift1[np.where(mindeltasA < 0.01)]
    Ax = np.ones(len(argA))
    argB = shift1[np.where(mindeltasB < 0.01)]
    Bx = np.zeros(len(argB))

    plotting.niceLinPlot(ax, dcps, shift1, logx=False, logy=False, color='r',
                        linestyle="", marker='o', markersize=1)

    plt.plot(-1.602*Ax, argA, linestyle="", marker='o', color='goldenrod')
    plt.plot(Bx, argB, linestyle="", marker='o', color='lightseagreen')
    ax.tick_params(which='both', direction="inout")
    ax.tick_params(which='both', top=True, right=True)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


    mindeltasA = np.absolute(dcps + 1.602)
    mindeltasB = np.absolute(dcps)
    argA = shift2[np.where(mindeltasA < 0.01)]
    print(argA)
    Ax = np.ones(len(argA))
    argB = shift2[np.where(mindeltasB < 0.01)]
    Bx = np.zeros(len(argB))


    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plotting.niceLinPlot(ax, dcps, shift2, logx=False, logy=False, color='b',
                         linestyle="", marker='o', markersize=1)
    ax.tick_params(which='both', direction="inout")
    ax.tick_params(which='both', top=True, right=True)
    plt.plot(-1.602*Ax, argA, linestyle="", marker='o', color='goldenrod')
    plt.plot(Bx, argB, linestyle="", marker='o', color='lightseagreen')

    plt.legend()
    plt.savefig('../images/' + outputFile)

if __name__ == '__main__':
    main()