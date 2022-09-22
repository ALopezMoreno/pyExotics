import numpy as np
import matplotlib.pyplot as plt
import time
from vanilla import oscillatorBase
import experiments
from graphing import plotting
from LED import KKmodes
import os
from matplotlib import ticker
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

parula = plotting.parula_map

L = 295.2 #810
Lnear = 0.280
profile='T2K'

start_time = time.time()
E = 1
#P1 = 0.935170599512747
#P2 = 0.2246213970198935
#P3 = 0.05041696464420451
#P4 = 0.05949550010437106

nbins = 80
mDirac = 0.5 #  Ev

exp = experiments.experiment(experiments.eProfile(profile), L, smooth=100)

Rs = np.linspace(0.0001, 1, 100)
# Rs = [0.0001]

for k in range(len(Rs)):
    if Rs[k]*mDirac < 0.05: #the approximation is good enough for small R*mDirac
        flag = True
    else:
        flag = False
    if os.path.exists('movie/comparison_'+str(k)+'.png'):
        continue
    # %%

    print('beginning new frame')
    print(k)
    propagator = KKmodes.KKoscillator(L, E, KKmodes.KKtower(10, mDirac, Rs[k], inverted=False, approx=flag))
    #propagator = oscillatorBase.Oscillator(L, E)
    #propagator.dcp = 2.618
    #propagator.setPMNS() #  We have updated the oscparams so we should set the pmns

    P1L = exp.propagate(propagator, 0, 0)
    P2L = exp.propagate(propagator, 1, 1)
    P3L = exp.propagate(propagator, 1, 0)
    P4L = exp.propagate(propagator, 1, 0, antineutrino=True)

    exp = experiments.experiment(experiments.eProfile(profile), Lnear, smooth=100)

    P1S = exp.propagate(propagator, 0, 0)
    P2S = exp.propagate(propagator, 1, 1)
    P3S = exp.propagate(propagator, 1, 0)
    P4S = exp.propagate(propagator, 1, 0, antineutrino=True)


    P1 = P1L #/ P1S
    P2 = P2L #/ P2S
    P3 = P3L #/ P2S
    P4 = P4L #/ P2S

    asimov_12 = 0.307
    asimov_23 = 0.561
    asimov_13 = 0.022
    asimov_cp =-1.602

    range_sin2th12 = [0,1]
    range_sin2th23 = [0,1]
    range_sin2th13 = [0,1]
    range_dcp = [-np.pi, np.pi]

    exp = experiments.experiment(experiments.eProfile(profile), L)
    propagator = oscillatorBase.Oscillator(L, E, smearing=[0, 0], inverted=0)  # Basic propagator for feeding it to experiment

    sin2th12_prop = np.linspace(range_sin2th12[0], range_sin2th12[1], nbins)
    sin2th23_prop = np.linspace(range_sin2th23[0], range_sin2th23[1], nbins)
    sin2th13_prop = np.linspace(range_sin2th13[0], range_sin2th13[1], nbins)
    dcp_prop = np.linspace(range_dcp[0], range_dcp[1], nbins)

    diffs = np.ones((nbins,nbins))

    time_now = time.time()
    exp.smooth=False

    for i in range(nbins):
        for j in range(nbins):
            #propagator.theta12 = np.arcsin(np.sqrt(sin2th12_prop[j]))
            propagator.theta23 = np.arcsin(np.sqrt(sin2th23_prop[i]))
            #propagator.theta13 = np.arcsin(np.sqrt(sin2th13_prop))
            propagator.dcp = dcp_prop[j]
            propagator.setPMNS()

            Pfit1 = exp.propagate(propagator, 0, 0)
            Pfit2 = exp.propagate(propagator, 1, 1)
            Pfit3 = exp.propagate(propagator, 1, 0)
            Pfit4 = exp.propagate(propagator, 1, 0, antineutrino=True)

            diffs1 = np.sum((Pfit1 - P1) ** 2) #/ np.sum(P1) ** 2
            diffs2 = np.sum((Pfit2 - P2) ** 2) #/ np.sum(P2) ** 2
            diffs3 = np.sum((Pfit3 - P3) ** 2) #/ np.sum(P3) ** 2
            diffs4 = np.sum((Pfit4 - P4) ** 2) #/ np.sum(P4) ** 2

            diffs[i, j] = 1 / (diffs1 + diffs2 + diffs3 + diffs4)

        print([i, time.time()-time_now], end='\r')
    #%%
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(5.5, 4.42))

    Y = sin2th23_prop
    X = dcp_prop

    cont = plotting.plot2Dcontour(ax, X, Y, diffs, logx=False, logy=False, percentile=[8], reverse=False)#     locator=ticker.LogLocator())
    #                                   j               i
    fig.suptitle('R = ' + str(round(Rs[k], 4)) + '    ' + r'$1/eV^2$')
    ax.set_box_aspect(1)
    #cbar = fig.colorbar(cont)
    bestfit = np.unravel_index(diffs.argmin(), diffs.shape)
    #print(bestfit)

    #ax.plot(Y[bestfit[0]], X[bestfit[1]], color='r', marker='+', markersize=8)
    ax.plot(asimov_cp, asimov_23, color='black', marker='o', markersize=2.5)
    ax.set_ylabel(r"$\sin^2\theta_{23}$", fontsize=15)
    ax.set_xlabel(r"$\delta_{CP}$", fontsize=15)
    ax.set_title("normal ordering", fontsize=12, loc='right')
    ax.xaxis.set_major_locator(ticker.LinearLocator(7))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    #ax.xaxis.set_minor_locator(ticker.LinearLocator(11 * 6 - 5))
    ax.yaxis.set_major_locator(ticker.LinearLocator(7))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f')) #'%.3f' .2e
    #ax.yaxis.set_minor_locator(ticker.LinearLocator(11 * 6 - 5))
    #ax.ticklabel_format(axis='y', style='sci')
    ax.tick_params(which='major', direction='out', top=True, right=True, labelsize=10)
    ax.tick_params(which='minor', direction='out', top=True, right=True)
    ax.grid(color='white', linestyle='dashed', linewidth=0.5, alpha=0.3)
    plt.savefig('movie/comparison_'+str(k)+".png")
    plt.close()
#plt.show()
