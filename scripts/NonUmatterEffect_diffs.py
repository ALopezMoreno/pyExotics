import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from graphing import plotting
from itertools import permutations
import matplotlib.ticker as ticker
import experiments

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})




## PARAMETERS ##
S12 = 0.307
S23 = 0.022
S13 = 0.561
dcp =-1.601

S14 = 0.
S24 = 0.
S34 = 0.
H1 = 0.0
mSterile = 10 ** 4

density = 2.7 / 2 # 2.7  # EARTH CRUST UNITS
G_f = 5.3948e-5
L = 295

fluxes = experiments.eProfile('T2K')
counts_ee = np.asarray(fluxes.nue)
binEdges_ee = np.asarray(fluxes.nueBE)
counts_mu = np.asarray(fluxes.numu)
binEdges_mu = np.asarray(fluxes.numuBE)

n = 10**3

energies = np.logspace(-2, 1, n)
#%%
## DERIVED PARAMETERS ##
th12 = np.arcsin(np.sqrt(S12))
th23 = np.arcsin(np.sqrt(S23))
th13 = np.arcsin(np.sqrt(S13))
th14 = np.arcsin(np.sqrt(S14))
th24 = np.arcsin(np.sqrt(S24))
th34 = np.arcsin(np.sqrt(S34))


A = np.asarray([[np.cos(th14), 0, 0],
                [-np.sin(th14)*np.sin(th24), np.cos(th24), 0],
                [-np.sin(th34)*np.cos(th24)*np.sin(th14), -np.sin(th24)*np.sin(th34), np.cos(th34)]], dtype=complex)


def matterHamiltonian_nonU(energy, density, ngens=3, earthCrust=False, neOverNa=False, electronDensity=False):

    global A
    # Take care of the units
    if earthCrust:
        G_f = 5.3948e-5
    elif neOverNa:
        G_f = 5.4489e-5
    elif electronDensity:
        G_f = 9.93e-2
    else:
        G_f = 1.166e-5

    #  nominal matter hamiltonian
    H = np.zeros((ngens, ngens))
    A_CC = np.sqrt(2) * G_f * energy * 2 * density
    A_NC = - np.sqrt(2) * G_f * energy * density
    H[0, 0] = A_CC + A_NC
    H[1, 1] = A_NC
    H[2, 2] = A_NC
    if ngens > 3:
        for i in range(3, ngens):
            H[i, i] = 1/2 * H[0, 0]

    prod = np.matmul(A, A.transpose())
    matterH = np.matmul(prod, H)
    return np.matmul(matterH, prod)


def getOsc(mix, eigs, alpha, beta, energy):
    global A
    global L
    pref = np.matmul(A, A.transpose().conjugate())
    part1 = mix[alpha, 0] * mix[beta, 0].conjugate() * mix[alpha, 1].conjugate() * mix[beta, 1]
    part2 = mix[alpha, 0] * mix[beta, 0].conjugate() * mix[alpha, 2].conjugate() * mix[beta, 2]
    part3 = mix[alpha, 1] * mix[beta, 1].conjugate() * mix[alpha, 2].conjugate() * mix[beta, 2]

    phase1 = (eigs[1] - eigs[0]) * L * 1.27 * 2 / (energy * 2)
    phase2 = (eigs[2] - eigs[0]) * L * 1.27 * 2 / (energy * 2)
    phase3 = (eigs[2] - eigs[1]) * L * 1.27 * 2 / (energy * 2)

    osc = np.absolute(pref[alpha, beta])**2 - 4 * (part1*np.sin(phase1)**2 +
                                                   part2*np.sin(phase2)**2 +
                                                   part3*np.sin(phase3)**2).real

    osc += 2 * (part1*np.sin(phase1*2) +
                part2*np.sin(phase2*2) +
                part3*np.sin(phase3*2)).imag

    return(osc)

prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                              1, False, False, 0, ngens=3)
U = prop.PMNS
N = np.matmul(A, U)

## Vanilla ##
vanil_ee = np.zeros(n)
vanil_emu = np.zeros(n)
vanil_mumu = np.zeros(n)

for i, energy in tqdm(enumerate(energies)):
    prop.update_hamiltonian(energy, density, ngens=3, earthCrust=True)
    vanil_ee[i] = prop.getOsc(0, 0)
    vanil_emu[i] = prop.getOsc(0, 1)
    vanil_mumu[i] = prop.getOsc(1, 1)


## True Non_U ##
nonU_ee = np.zeros(n)
nonU_emu = np.zeros(n)
nonU_mumu = np.zeros(n)

prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                              1, False, False, 0, ngens=4)
prop.generations=4
prop.masses.append(mSterile)

prop.mixingPars = [np.arcsin(np.sqrt(0.307)),
                   np.arcsin(np.sqrt(0.022)),
                   np.arcsin(np.sqrt(0.561)),
                   np.arcsin(np.sqrt(S14)),
                   np.arcsin(np.sqrt(S24)),
                   np.arcsin(np.sqrt(S34)),
                   -1.601, H1, H1]
prop.update()

for i, energy in tqdm(enumerate(energies)):
    elow  = energy*0.9999
    ehigh = energy*1.0001
    ens = np.linspace(elow, ehigh, 3)
    nonU_ee[i] = 0
    nonU_emu[i] = 0
    nonU_mumu[i] = 0

    for en in ens:
        prop.update_hamiltonian(en, density, ngens=4, earthCrust=True)
        nonU_ee[i] += prop.getOsc(0, 0) / 3.
        nonU_emu[i] += prop.getOsc(0, 1) / 3.
        nonU_mumu[i] += prop.getOsc(1, 1) / 3.

## CASE 1 ##

# Reset propagator
prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                              1, False, False, 0, ngens=3)

case1_ee = np.zeros(n)
case1_emu = np.zeros(n)
case1_mumu = np.zeros(n)

for i, energy in tqdm(enumerate(energies)):
    prop.update_hamiltonian(energy, density, ngens=3, earthCrust=True)
    prop.mixingMatrix = np.matmul(A, prop.mixingMatrix)

    case1_ee[i] = prop.getOsc( 0, 0)
    case1_emu[i] = prop.getOsc( 0, 1)
    case1_mumu[i] = prop.getOsc( 1, 1)



## CASE 2 ##

# Reset propagator
prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                              1, False, False, 0, ngens=3)

prop.autoupdate = False

case2_ee = np.zeros(n)
case2_emu = np.zeros(n)
case2_mumu = np.zeros(n)

for i, energy in tqdm(enumerate(energies)):
    prop.PMNS = np.matmul(A, prop.PMNS)
    prop.update_hamiltonian(energy, density, ngens=3, earthCrust=True)
    mix = prop.mixingMatrix
    eigs = prop.eigenvals

    case2_ee[i] = prop.getOsc(0,0) #getOsc(mix, eigs, 0,0,energy)
    case2_emu[i] = prop.getOsc(0, 1)
    case2_mumu[i] = prop.getOsc(1, 1)
    prop.PMNS = U

## CASE 3 ##

# Reset propagator
prop = customPropagator.HamiltonianPropagator(matterHamiltonian_nonU, L,
                                              1, False, False, 0, ngens=3)

prop.autoupdate = False

case3_ee = np.zeros(n)
case3_emu = np.zeros(n)
case3_mumu = np.zeros(n)

for i, energy in tqdm(enumerate(energies)):
    #prop.PMNS = N
    prop.update_hamiltonian(energy, density, ngens=3, earthCrust=True)
    mix = prop.mixingMatrix
    eigs = prop.eigenvals
    prop.mixingMatrix = np.matmul(A, prop.mixingMatrix)

    #case3_ee[i] = getOsc(mix, eigs, 0,0,energy)
    case3_ee[i] = prop.getOsc(0, 0)
    case3_emu[i] = prop.getOsc(0, 1)
    case3_mumu[i] = prop.getOsc(1, 1)


#%%
## GET PROGPU PROBS ##
data1 = np.loadtxt("../oscillationProbs/vanilla.txt")
data2 = np.loadtxt("../oscillationProbs/probGpu_nonU_un-normalised.txt")
#data3 = np.loadtxt("../oscillationProbs/apparentNonU.txt")
data3 = np.loadtxt("../oscillationProbs/probGpu_nonU_normalised.txt")

probs1 = data1[:, 1]
ens1 = data1[:, 0]

probs2 = data2[:, 1]
ens2 = data2[:, 0]

probs3 = data3[:, 1]
ens3 = data3[:, 0]

diff_HNL_probGpu = (probs2 - nonU_ee) * 100


## PLOTTING ##

fig, ax = plt.subplots(2, 1, figsize=(5.5, 5.5), dpi=200, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax[0].bar(x=binEdges_ee[:-1], height=counts_ee/np.max(counts_ee * 2/2), width=np.diff(binEdges_ee),
       align='edge', fc='blue', alpha=0.3, label=r'$\Phi_e(ND)$')

#ax[0].bar(x=binEdges_mu[:-1], height=counts_mu/np.max(counts_mu * 2/2), width=np.diff(binEdges_mu),
#       align='edge', fc='green', alpha=0.3, label=r'$\Phi_\mu(ND)$')

plotting.niceLinPlot(ax[0], energies, vanil_ee, logx=True, logy=False, color='black', linewidth=1, label=r'$3\nu$ num')
#plotting.niceLinPlot(ax, energies, case1_emu, logx=True, logy=False, color='red', linewidth=1)
#plotting.niceLinPlot(ax, energies, case2_emu, logx=True, logy=False, color='lime', linewidth=1)
#plotting.niceLinPlot(ax, energies, case3_emu, logx=True, logy=False, color='blue', linewidth=1)
plotting.niceLinPlot(ax[0], energies, nonU_ee, logx=True, logy=False, color='gold', linewidth=1, label=r'$4\nu$ num')

#plotting.niceLinPlot(ax, ens1, probs1, logx=True, logy=False, color='black', linewidth=1,
#                     linestyle='-.')
plotting.niceLinPlot(ax[0], ens3, probs3, logx=True, logy=False, color='r', linewidth=1,
                     linestyle='-.', label=r'$4\nu^*$ pGpu')

plotting.niceLinPlot(ax[0], ens2, probs2, logx=True, logy=False, color='blue', linewidth=1,
                     linestyle='-.', label=r'$4\nu \;$ pGpu')

plotting.niceLinPlot(ax[1], energies, diff_HNL_probGpu, logx=True, logy=False, color='blue', linewidth=1)
ax[1].axhline(y=0, linewidth=1, color='black')
ax[1].set_ylim(-1, 1)

plotting.makeTicks(ax[0], energies)
plotting.makeTicks(ax[1], ynumber=4)

ax[1].set_yticks([-0.5, 0.5])

ax[1].set_xlabel(r'E (GeV)', fontsize=20)
ax[1].set_ylabel(r'$\Delta \%$', fontsize=20)
ax[0].set_ylabel(r'$P(\nu_e\to\nu_e)$', fontsize=20)
plt.subplots_adjust(hspace=0)
ax[0].legend(fontsize=15)
plt.tight_layout()


plt.savefig('../images/NonU_probGpu_comparison_Pee.png')
plt.show()