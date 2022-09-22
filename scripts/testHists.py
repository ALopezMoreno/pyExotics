import numpy as np
import matplotlib.pyplot as plt
import time
from vanilla import oscillatorBase
import experiments
from graphing import plotting
from nonUnitary import sterileOsc
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

parula = plotting.parula_map

L = 295.2  # 810
Lnear = 0.280  # 1
profile = 'T2K'

start_time = time.time()

nbins = 10
mDirac = 0.5  # Ev

baseEprofile = experiments.eProfile(profile)

# Get bin centres:
# x1 = baseEprofile.nueBE[:-1] + (baseEprofile.nueBE[1:] - baseEprofile.nueBE[:-1] / 2)
# x2 = baseEprofile.numuBE[:-1] + (baseEprofile.numuBE[1:] - baseEprofile.numuBE[:-1] / 2)
# x3 = baseEprofile.numuBE[:-1] + (baseEprofile.numuBE[1:] - baseEprofile.numuBE[:-1] / 2)
# x4 = baseEprofile.numu_barBE[:-1] + (baseEprofile.numu_barBE[1:] - baseEprofile.numu_barBE[:-1] / 2)

exp = experiments.experiment(baseEprofile, L, smooth=100)
propagator = oscillatorBase.Oscillator(L, 1)

# Get nominal oscilated spectra
P1L = exp.propagate(propagator, 0, 0)
P2L = exp.propagate(propagator, 1, 1)
P3L = exp.propagate(propagator, 1, 0)
P4L = exp.propagate(propagator, 1, 0, antineutrino=True)

exp.L = 0.280
P1S = exp.propagate(propagator, 0, 0)
P2S = exp.propagate(propagator, 1, 1)
P3S = exp.propagate(propagator, 1, 0)
P4S = exp.propagate(propagator, 1, 0, antineutrino=True)

# FIX THIS WITH MORE EFFICIENT BOy
for i in range(len(P1S)):
    if P1S[i] < 1:
        P1S[i] = 1

for i in range(len(P2S)):
    if P2S[i] < 1:
        P2S[i] = 1

P1_0 = P1L / P1S
P2_0 = P2L / P2S
P3_0 = P3L / P2S
P4_0 = P4L / P2S

#range_dcp = [-np.pi, np.pi]
range_R = [0.0001, 1]
frames = 200

#range of non-unitary parametrs
range_14 = [0, 1]
range_24 = [0, 1]
range_34 = [0, 1]

range_p14 = [0, 2*np.pi]
range_p24 = [0, 2*np.pi]
range_p34 = [0, 2*np.pi]

#R = np.linspace(range_R[0], range_R[1], frames)
sin2th14 = np.linspace(range_14[0], range_14[1], frames)
sin2th24 = np.linspace(range_24[0], range_24[1], frames)
phi24 = np.linspace(range_p24[0], range_p24[1], frames)
for i in range(frames + 80):
    #if mDirac*R[i] < 0.05:
    #    flag = True
    #else:
    #    flag = False
    if os.path.exists('/home/andres/Desktop/pyExotics/movieHist/comparison_' + str(i) + ".png"):
        continue
    #propagator = KKmodes.KKoscillator(L, 1, KKmodes.KKtower(10, mDirac, R[i], inverted=False, approx=flag))
    propagator = sterileOsc.oneHNL_oscillator(L, 1)
    exp.L = 295.2

    propagator.theta14 = np.arcsin(np.sqrt(sin2th14[np.min([i, 40])]))
    propagator.theta24 = np.arcsin(np.sqrt(sin2th24[np.min([np.max([i - 40, 0]), 40])]))
    propagator.phi24 = phi24[np.max([i - 80], 0)]

    propagator.set_nonUnitary()
    propagator.build_mixing_matrix()

    # Propagate for the new setup
    Posc1 = exp.propagate(propagator, 0, 0)
    Posc2 = exp.propagate(propagator, 1, 1)
    Posc3 = exp.propagate(propagator, 1, 0)
    Posc4 = exp.propagate(propagator, 1, 0, antineutrino=True)

    # Update experimental setup for calculating SBL probs
    exp.L = 0.280
    Ps1 = exp.propagate(propagator, 0, 0)
    Ps2 = exp.propagate(propagator, 1, 1)

    for p in range(len(Ps1)):
        if Ps1[p] < 1:
            Ps1[p] = 1

    for p in range(len(Ps2)):
        if Ps2[p] < 1:
            Ps2[p] = 1

    # Get correct ratio
    Posc1 = Posc1 / Ps1
    Posc2 = Posc2 / Ps2
    Posc3 = Posc3 / Ps2
    Posc4 = Posc4 / Ps2

    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(8, 8))
    ax[0, 0].bar(x=baseEprofile.nueBE[:-1], height=P1_0, width=np.diff(baseEprofile.nueBE),
                 align='edge', fc='blue', ec='none', alpha=0.5)
    ax[0, 1].bar(x=baseEprofile.numuBE[:-1], height=P2_0, width=np.diff(baseEprofile.numuBE),
                 align='edge', fc='blue', ec='none', alpha=0.5)
    ax[1, 0].bar(x=baseEprofile.numuBE[:-1], height=P3_0, width=np.diff(baseEprofile.numuBE),
                 align='edge', fc='blue', ec='none', alpha=0.5)
    ax[1, 1].bar(x=baseEprofile.numu_barBE[:-1], height=P4_0, width=np.diff(baseEprofile.numu_barBE),
                 align='edge', fc='blue', ec='none', alpha=0.5)


    ax[0, 0].bar(x=baseEprofile.nueBE[:-1], height=Posc1, width=np.diff(baseEprofile.nueBE),
                 align='edge', fc='red', ec='none', alpha=0.5)
    ax[0, 1].bar(x=baseEprofile.numuBE[:-1], height=Posc2, width=np.diff(baseEprofile.numuBE),
                 align='edge', fc='red', ec='none', alpha=0.5)
    ax[1, 0].bar(x=baseEprofile.numuBE[:-1], height=Posc3, width=np.diff(baseEprofile.numuBE),
                 align='edge', fc='red', ec='none', alpha=0.5)
    ax[1, 1].bar(x=baseEprofile.numu_barBE[:-1], height=Posc4, width=np.diff(baseEprofile.numu_barBE),
                 align='edge', fc='red', ec='none', alpha=0.5)

    ax[0, 0].set_title(r'$\nu_e\rightarrow\nu_e$', loc='right')
    ax[0, 1].set_title(r'$\nu_\mu\rightarrow\nu_\mu$', loc='right')
    ax[1, 0].set_title(r'$\nu_\mu\rightarrow\nu_e$', loc='right')
    ax[1, 1].set_title(r'$\bar{\nu_\mu}\rightarrow\bar{\nu_e}$', loc='right')
    fig.suptitle(r"$\sin^2\theta_{14}=$" + ' ' + str(round(sin2th14[np.min([i, 40])], 3)) + ';     ' +
                 r"$\sin^2\theta_{24}=$" + ' ' + str(round(sin2th24[np.min([np.max([i - 40, 0]), 40])], 3)) + ';     '+
                 r"$\phi_{24}=$" + ' ' + str(round(phi24[np.max([i -80 , 0])], 3)), fontsize=17)
    for a in ax:
        for b in a:
            b.set_ylim(0, 1)

    for k in range(2):
        ax[0, k].set_xlim(0.01, 10)
        ax[1, k].set_xlim(0.01, 10)
        for j in range(2):
            ax[k, j].set_xlabel('E (GeV)')
            ax[k, j].set_ylabel('Flux ratios')
            #ax[k, j].set_yticks([0, 1])
            ax[k, j].set_box_aspect(1)
            ax[k, j].set_xscale('log')

    plt.tight_layout()
    plt.savefig('/home/andres/Desktop/pyExotics/movieHist/comparison_' + str(i) + ".png")
    plt.close()
    print('done %i of %i' % (i+1, frames), end='\r')
