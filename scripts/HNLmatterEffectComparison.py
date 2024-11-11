##  This script is used to compare three methods of generating the oscillation probabilities in the region pertinent
##  4nu hamiltonian solver, HNL calculation, and T2K approximation

import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from graphing import plotting
from HamiltonianSolver import customPropagator
import tqdm
np.set_printoptions(precision=3)
from matplotlib import ticker
import experiments

def get_A(S14, S24, S34, d14, d24):
    th14 = np.arcsin(np.sqrt(S14))
    th24 = np.arcsin(np.sqrt(S24))
    th34 = np.arcsin(np.sqrt(S34))

    A = np.asarray([[np.cos(th14), 0, 0],
                    [-np.sin(th14) * np.sin(th24) * np.exp(1j*(d24-d14)), np.cos(th24), 0],
                    [-np.sin(th34) * np.cos(th24) * np.sin(th14) * np.exp(-1j*d14), -np.sin(th24) * np.sin(th34) * np.exp(-1j*d24), np.cos(th34)]],
                    dtype=complex)

    return A

def matterHamiltonian_HNL(energy, density, ngens=3, earthCrust=False, neOverNa=False, electronDensity=False):

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


e = {'index':0, 'label':r'$\nu_e$'}
mu = {'index':1, 'label':r'$\nu_\mu$'}
tau = {'index':2, 'label':r'$\nu_\tau$'}


#%% SET PARAMETER VALUES##

# Non unitary parameters
S14 = 0
S24 = 0.2
S34 = 0.
d14 = 0 #np.pi/3
d24 = 0 #np.pi/6 # this is the invisible phase

# Unitary parameters
S12 = 0.307
S13 = 0.018
S23 = 0.561
dcp = -1.601

# Additional masses
mSterile = 10 ** 8

# Propagation parameters
density = 2.7 / 2  # 2.7  # EARTH CRUST UNITS
G_f = 5.3948e-5
L = 295

# Plotting parameters
n = 500
energies = np.logspace(-1.3, 1, n)

channels = [[mu, e], [mu, mu]]
colors = ['orange', 'orange']
t2kcolors = ['red', 'red']
numcolors = ['blue', 'blue']
stl = ['solid','dashdot']
label = [r'$\nu_\mu\to\nu_e$', r'$\nu_\mu\to\nu_\mu$']
numlabel = [r'Num 4$\nu$', None]
t2klabel = [r'T2K', None]
nu3label = [r'Num 3$\nu$', None]
#%% GENERATE NECESSARY MATRICES AND INSTANCE NECESSARY OBJECTS

# Non-unitary matrix
A = get_A(S14, S24, S34, d14, d24)

# 4-flavour numerical propagator
prop_num = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                               1, False, False, 0, ngens=4)

# HNL formalism numerical propagator
prop_HNL = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                               1, False, False, 0, ngens=3)

# T2K 3-flavour matter effect + HNL formalism approximation propagator
prop_T2K = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                               1, False, False, 0, ngens=3)

prop_3nu = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L,
                                               1, False, False, 0, ngens=3)



# Arrays to save probabilities in
probs_num = np.zeros(n)
probs_HNL = np.zeros(n)
probs_T2K = np.zeros(n)
probs_3nu = np.zeros(n)


#%% SET UP PROPAGATOR OBJECTS
print('setting up propagators \n')
# Update numerical propagator to 4 flavours
prop_num.generations = 4
prop_num.masses.append(mSterile)

prop_num.mixingPars = [np.arcsin(np.sqrt(S12)),
                       np.arcsin(np.sqrt(S13)),
                       np.arcsin(np.sqrt(S23)),
                       np.arcsin(np.sqrt(S14)),
                       np.arcsin(np.sqrt(S24)),
                       np.arcsin(np.sqrt(S34)),
                       dcp, d14, d24]  # I MIGHT HAVE DEFINED THE PHASES DIFFERENTLY

prop_HNL.mixingPars = [np.arcsin(np.sqrt(S12)),
                       np.arcsin(np.sqrt(S13)),
                       np.arcsin(np.sqrt(S23)),
                       dcp]

prop_T2K.mixingPars = [np.arcsin(np.sqrt(S12)),
                       np.arcsin(np.sqrt(S13)),
                       np.arcsin(np.sqrt(S23)),
                       dcp]

prop_3nu.mixingPars = [np.arcsin(np.sqrt(S12)),
                       np.arcsin(np.sqrt(S13)),
                       np.arcsin(np.sqrt(S23)),
                       dcp]


prop_num.update()
prop_HNL.update()
prop_T2K.update()
prop_3nu.update()

prop_HNL.autoupdate = False
prop_T2K.autoupdate = False
prop_3nu.autoupdate = False

# Unitary part of the mixing matrix
U = prop_3nu.PMNS

# Complete mixing matrix
N = np.matmul(A, U)
# Don't autoupdate the HNL and T2K propagators because the PMNS matrix will be modified

#prop_HNL.PMNS = N

# Set up plotting object
fig, ax = plt.subplots(2, 1, sharex=True, dpi=300, figsize=(5, 5), gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})
ax[1].axhline(0, color='black',linestyle='dashed')

# Add NDflux
fluxes = experiments.eProfile('T2K')
counts_mu = np.asarray(fluxes.numu)
binEdges_mu = np.asarray(fluxes.numuBE)

ax[0].bar(x=binEdges_mu[:-1], height=counts_mu/np.max(counts_mu * 2.25/2), width=np.diff(binEdges_mu),
       align='edge', fc='green', alpha=0.3, label=r'$\Phi_\mu(ND)$')
#%% GET OSCILLATION PROBABILITIES
for k, flavours in enumerate(channels):
    for i, energy in tqdm.tqdm(enumerate(energies)):
        prop_num.update_hamiltonian(energy, density, ngens=4, earthCrust=True, Ndensity=50*density)
        probs_num[i] = prop_num.getOsc(flavours[0]['index'], flavours[1]['index'])

        prop_HNL.update_hamiltonian(energy, density, ngens=3, earthCrust=True, Ndensity=50*density, N=A)
        prop_HNL.mixingMatrix = np.matmul(A, prop_HNL.mixingMatrix)
        probs_HNL[i] = prop_HNL.getOsc(flavours[0]['index'], flavours[1]['index'])
        prop_HNL.mixingMatrix = np.matmul(A, prop_HNL.mixingMatrix)

        prop_T2K.update_hamiltonian(energy, density, ngens=3, earthCrust=True, Ndensity=50*density)
        prop_T2K.mixingMatrix = np.matmul(A, prop_T2K.mixingMatrix)
        probs_T2K[i] = prop_T2K.getOsc(flavours[0]['index'], flavours[1]['index'])

        prop_3nu.update_hamiltonian(energy, density, ngens=3, earthCrust=True, Ndensity=50*density)
        probs_3nu[i] = prop_3nu.getOsc(flavours[0]['index'], flavours[1]['index'])



#%% PLOT

    plotting.niceLinPlot(ax[0], energies, probs_num, logy=False, color=numcolors[k],  linewidth=1.5, alpha=.7, linestyle=stl[k], label=numlabel[k])
    # MY METHOD DOESN'T WORK CAUSE IT ALWAYS TRIES TO FIND A UNITARY DECOMPOSITION MATRIX.
    plotting.niceLinPlot(ax[0], energies, probs_HNL, logy=False, color='orange', linestyle='dotted')
    plotting.niceLinPlot(ax[0], energies, probs_T2K, logy=False, color=t2kcolors[k],  linewidth=1.5, alpha=.7, linestyle=stl[k], label=t2klabel[k])
    plotting.niceLinPlot(ax[0], energies, probs_3nu, logy=False, color='black', linewidth=1.5, alpha=.5, linestyle=stl[k], label=nu3label[k])
    plotting.niceLinPlot(ax[1], energies, probs_num-probs_T2K, logy=False, color=colors[k],  linewidth=1.5, alpha=1, linestyle=stl[k], label=label[k])

# Make pretty
ax[0].set_ylim(0, 1)
ax[1].set_ylim(-0.02, 0.02)
ax[0].set_xlim(energies[0],energies[-1])
for axis in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[axis].set_linewidth(1.5)
    ax[1].spines[axis].set_linewidth(1.5)


        #ax.set_box_aspect(1)
ax[1].yaxis.set_major_locator(ticker.LinearLocator(5))
ax[1].yaxis.set_minor_locator(ticker.LinearLocator(21))
ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))


ax[0].tick_params(axis='both', which='major', labelsize=15, direction='in', length=5, width=1, top=True, right=True)
ax[0].tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)
ticks = ax[0].yaxis.get_major_ticks()
if len(ticks) > 0:
    ticks[0].set_visible(False)  # Remove the first major tick
    ticks[-1].set_visible(False)  # Remove the last major tick

ax[1].tick_params(axis='x', which='major', labelsize=15, direction='inout', length=10, width=1, top=True, right=True)
ax[1].tick_params(axis='y', which='major', labelsize=15, direction='in', length=10, width=1, top=True, right=True)
ax[1].tick_params(axis='y', which='minor', direction='in', length=4, width=1, top=True, right=True)
ax[1].tick_params(axis='x', which='minor', direction='in', length=3, width=1, top=True, right=True)
ax[0].tick_params(axis='x', which='minor', direction='in', length=3, width=1, top=True, right=True)
ax[1].set_yticklabels([None, r'$-1\%$', None, r'$+1\%$', None])
ax[1].legend( loc='upper right', fontsize=11)  # Adjust loc and ncol as
ax[0].legend( loc='lower right', fontsize=11)

ax[1].set_xlabel(r'E (GeV)', fontsize=15)
ax[1].set_ylabel(r'$\Delta$P', fontsize=15)
ax[0].set_ylabel(r'P$(\nu_\alpha\to\nu_\beta)$', fontsize=15)

fig.savefig('../images/matterEffectComparison.pdf', format='pdf', bbox_inches='tight')
plt.show()