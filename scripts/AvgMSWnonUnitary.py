import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from graphing import plotting
from itertools import permutations
from solarFitting import LMAmodels
import matplotlib

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})
matplotlib.rcParams.update({
    'font.size': 14,          # Default font size
    'axes.labelsize': 16,     # Font size of axis labels
    'axes.titlesize': 18,     # Font size of plot title
    'xtick.labelsize': 18,    # Font size of x-axis tick labels
    'ytick.labelsize': 18,    # Font size of y-axis tick labels
    'legend.fontsize': 15     # Font size of legend
})


energyMin = np.log10(0.12)
energyMax = np.log10(20)
nEnergies = 200
nGens = 4

S14 = 0.1
S24 = 0.0
S34 = 0.8
Hij = 0

N_e = 103
N_n = 103/2.3

th14 = np.arcsin(np.sqrt(S14))
th24 = np.arcsin(np.sqrt(S24))
th34 = np.arcsin(np.sqrt(S34))
#(np.cos(th14))
#print(np.sin(th14))
# Make list of energies to loop through
energies = np.logspace(energyMin, energyMax, nEnergies)

def LMA_solution_HNL(energy, th12, th13, a11, a31, a33):

    global N_n, N_e

    dm21 = 7.42 * 10 ** (-5)
    #beta = np.sqrt(2) * 5.3948e-5 * energy * 10 ** -3 / dm21 * (a11**4 * (2*N_e*np.cos(th13)**2) - a11**2*a33*a31*N_n*np.sin(2*th13)
    #                                                            - a11**2*a33**2*N_n*np.sin(th13)**2)

    # PAPER VERSION (WITH N_N)
    # beta = (np.sqrt(2) * 5.4489e-5 * energy * 10 ** -3 / (dm21) ) * (a11**2 * (2*N_e*np.cos(th13)**2 + N_n/2) - 2*N_e*a31*np.sin(2*th13))
    # beta = ( np.sqrt(2) * 5.4489e-5 * energy * 10 ** -3 / (dm21)) * (a11 ** 2 * (2*N_e) * np.cos(th13)**2 + a31 * a33 ** 2 * N_n * np.sin(th13)**2)
    beta = (np.sqrt(2) * 5.4489e-5 * energy * 10 ** -3 / (dm21)) * ( a11**2 * N_e * (1+np.cos(2*th13)) +0*a11*a31*np.cos(2*th13)*N_e ) #
            #(a11**2*2*N_e + (1-a11**2)*N_n) * np.cos(th13) ** 2  - N_n * (a33**2*np.sin(th13)**2 + a31*a33*np.sin(2*th13)) ) # SEEMS CORRECT!!!

    #beta =  (np.sqrt(2) * 5.4489e-5 * energy * 10 ** -3 / (dm21)) * (
    #        np.cos(th13)**2*(a11**2*2*N_e + (1-a11**2)*N_n) - N_n*(a33**4*np.sin(th13)**2 + np.sin(2*th13)*(a11*a31 + a31/a11*a33**2)) )
    #beta = (np.sqrt(2) * 5.4489e-5 * energy * 10 ** -3 / (dm21)) * (
    #        (a11**2*2*N_e + (1-a11**2)*N_n) * np.cos(th13) ** 2 - N_n * (a33**2*np.sin(th13)**2 + a31*a33*np.sin(2*th13)) ) # SEEMS CORRECT!!!



                                                                     #* np.cos(th13) + a33 ** 2 * N_n/2 * np.sin(th13)**2 )
                #a11 ** 2 * 2*(N_e * np.cos(th13)**2 ) )#2 * N_e * a11 * a31 * np.sin(2 * th13))
    # NEW VERSION
    #beta = np.sqrt(2) * 5.3948e-5 * energy * 10 ** -3 / dm21 * (a11**4 * ((2*N_e - N_n)*np.cos(th13)**2) - a11**2 * N_n * (a33*a31*np.sin(2*th13)) - a33**2*np.sin(th13)**2)


    matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
    probLMA = (np.cos(th13)**4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) + np.sin(th13)**4) * a11**4

    return probLMA

def makeProbs(energies, nGens):
    # Set up propagator
    matterHam = customPropagator.matterHamiltonian

    prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 1, 1, False, False, 0,
                                                  ngens=nGens)
    prop.generations = nGens
    global S14, S24, S34, Hij, N_e, N_n
    if nGens == 4:
        sterileMass = 10 ** 6
        # """
        prop.masses.append(sterileMass)

        prop.mixingPars = [np.arcsin(np.sqrt(0.307)),
                           np.arcsin(np.sqrt(0.022)), #0.022
                           np.arcsin(np.sqrt(0.561)),
                           np.arcsin(np.sqrt(S14)),
                           np.arcsin(np.sqrt(S24)),
                           np.arcsin(np.sqrt(S34)),
                           -1.601, Hij, Hij]

        prop.update()
        # """
    a = []
    #print(S14, S24, S34)
    matters = []
    for i, e in tqdm(enumerate(energies), total=nEnergies):
        prop.update_hamiltonian(e * 10 ** -3, N_e, ngens=nGens, neOverNa=True, Ndensity=N_n, N=prop.PMNS)
        prop.update()
        mixingMatrix = prop.mixingMatrix
        vacuumMatrix = prop.PMNS
        matt = prop.newHam
        # print(prop.newHam)
        P_ee = 0
        for k in range(3):
            P_ee += np.absolute(mixingMatrix[0, k].conjugate() * vacuumMatrix[0, k]) ** 2

        a.append(P_ee)
        matters.append(matt)
    return a, matters, prop.PMNS


a, matters, PMNS4 = makeProbs(energies, 4)
#ax[1].plot(energies, a, color='blue', alpha=0.6, linewidth=2, label='HNL (numerical)')

a0, matters0, PMNS = makeProbs(energies, 3)


fig, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=150, sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})


# Set logarithmic scales for x and y axes
ax[1].set_xscale('log')

# ax[1].plot(energies, a, color='blue', alpha=0.6, linewidth=2, label='HNL (numerical)')

# Set tick positions and labels
#ax[1].gca().xaxis.tick_top()
for i in range(2):
    ax[i].xaxis.set_label_position('bottom')
    ax[i].yaxis.tick_left()
    ax[i].yaxis.set_label_position('left')
    ax[i].minorticks_on()
    ax[i].xaxis.set_tick_params(which='both', direction='in', width=1, length=5)
    ax[i].yaxis.set_tick_params(which='both', direction='in', width=1, length=5)
    ax[i].xaxis.set_tick_params(which='minor', direction='in', width=1, length=2)
    ax[i].yaxis.set_tick_params(which='minor', direction='in', width=1, length=2)

# Increase the size of tick labels
#ax[1].xticks(fontsize=18)
#ax[1].yticks(fontsize=18)
#ax[1].ylim(0,0.6)

# Set axis labels
ax[1].set_xlabel(r'E (MeV)', fontsize=20)
ax[1].set_ylabel(r'$\Delta P_{ee}/P_{ee}$', fontsize=20)
ax[0].set_ylabel(r'$P_{ee}$', fontsize=20)
# Add a grid
#ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)


th12 = np.arcsin(np.sqrt(0.307))
th13 = np.arcsin(np.sqrt(0.022)) # 0.022
lma_condition = np.cos(2 * th12)
msw_condition = 1

dm12 = (7.53 * 10 ** (-5))
vacuum_2f = 1 - 0.5 * np.sin(2 * th12) ** 2

vac_prob = np.cos(th13) ** 4 * vacuum_2f + np.sin(th13) ** 4
msw_prob = np.sin(th12) ** 2 * np.cos(th13) ** 4

beta = (2 * np.sqrt(2) * 5.4489e-5 * np.cos(th13) ** 2 * 103 * energies * 10 ** -3) / (7.42 * 10 ** (-5))
matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
probsLMA = np.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) + np.sin(th13)**4


alpha = np.asarray([[np.cos(th14), 0, 0],
                    [-np.sin(th14)*np.sin(th24), np.cos(th24), 0],
                    [-np.sin(th34)*np.cos(th24)*np.sin(th14), -np.sin(th24)*np.sin(th34), np.cos(th34)]])

PMNS2 = np.matmul(alpha, PMNS)

#print(np.linalg.det(alpha))
#print(np.cos(th14)**2)

probVacuum = np.cos(th14)**4 - 4*(np.abs(PMNS2[0, 0])**2*np.abs(PMNS2[0,1])**2/2 +
                                  np.abs(PMNS2[0, 0])**2*np.abs(PMNS2[0,2])**2/2 +
                                  np.abs(PMNS2[0, 1])**2*np.abs(PMNS2[0,2])**2/2)

probsLMA2 = LMA_solution_HNL(energies, th12, th13, np.cos(th14), np.sin(th14)*np.sin(th34)*np.cos(th24), np.cos(th34))

#closest_index2 = np.argmin(np.abs(matters - crit2))
#closest_value2 = energies[closest_index2]


# ax[1].plot(energies, probsLMA2, color='blue', linewidth=2, linestyle='-.', label='HNL LMA')

vals = [1, 0.99, 0.95, 0.9, 0.85]
labels = [r'$\sin^2\theta_{i4}=0$', r'$\sin^2\theta_{i4}=0.02$', r'$\sin^2\theta_{i4}=0.1$', r'$\sin^2\theta_{i4}=0.2$', r'$\sin^2\theta_{i4}=0.27$']
labels1 = [r'HNL (numerical)', '', '', '', '']
labels2 = [r'HNL (approx.)',  '', '', '', '']
colors = ['black', 'blue', 'red', 'orange', 'limegreen']
for i, val in enumerate(vals):
    S14 = 1-val**2
    S24 = 1-val**2
    S34 = 1-val**2

    th14 = np.arcsin(np.sqrt(S14))
    th24 = np.arcsin(np.sqrt(S24))
    th34 = np.arcsin(np.sqrt(S34))

    a11 = np.cos(th14)
    a22 = np.cos(th24)
    a33 = np.cos(th34)
    a21 = np.sin(th24)*np.sin(th14)
    a31 = np.sin(th14)*np.sin(th34)*np.cos(th24)
    #print(a11**2, a33, a31)

    print('BINGO')
    a, matters, PMNS4 = makeProbs(energies, 4)
    #print(matters[-1])
    probsLMA2 = LMAmodels.LMA_solution_4nu(energies, dm12, 0.307,0.022, a11, N_e=N_e, N_n=N_n)
    #ax[1].plot(energies, (np.asarray(a) - probsLMA2)/np.asarray(a), color='red', alpha=0.6, linewidth=2)
    ax[1].plot(energies, (np.asarray(a)-probsLMA2) / np.asarray(a), color=colors[i], alpha=0.6, linewidth=2, label=labels[i])
    ax[0].plot(energies, np.asarray(a), color=colors[i], alpha=0.6, linewidth=2, label=labels1[i])
    ax[0].plot(energies, probsLMA2, color=colors[i], alpha=1, linewidth=2, linestyle='--', label=labels2[i])

    #ax[1].plot(energies, probsLMA2, color='red', alpha=0.6, linewidth=2)

    #ax[1].plot(energies, probsLMA2, color='blue', linewidth=2, linestyle='-.')

ax[1].set_ylim(-0.025,0.025)
ax[1].set_xlim(0.12, 20)
ax[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
#ax[1].plot(energies, a0, color='red', alpha=0.6, linewidth=2, label=r'3$\nu$ (numerical)')
#ax[1].plot(energies, probsLMA, color='purple', linewidth=2, linestyle='-.', label=r'3$\nu$ LMA')

#ax[1].axhline(y=probVacuum, color='blue')
#ax[1].axhline(y=msw_prob, color='red', xmin=closest_index/nEnergies)

#ax[1].axhline(y=vac_prob * (1-S14)**2, color='green', xmax=closest_index/nEnergies)
#ax[1].axhline(y=msw_prob - 1+(1-S14), color='red', xmin=closest_index/nEnergies)

#ax[1].axvline(x=closest_value)
ax[1].legend(loc='upper right')
ax[1].legend(ncol=2)
ax[0].set_ylim(0.1,0.8)
ax[0].legend(loc='upper right')
#fig.suptitle('LMA electron survival probabilities', fontsize=20)
plt.subplots_adjust(hspace=0)
for ax2 in ax:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(1.5)
fig.savefig('../images/LMAapproximation.pdf', format='pdf')
plt.show()

"""
numbers = np.arange(6)
desired_permutations = []
for perm in permutations(numbers):
    for i in range(4):  # Check for 2, 1, 0 in adjacent positions
        if perm[i] == 2 and perm[i + 1] == 1 and perm[i + 2] == 0:
            desired_permutations.append(perm)
            break

# Convert the list of tuples to a numpy array
permutations_array = np.array(desired_permutations)



energies = [10**-2]
values = []
for ordering in permutations_array:
    a, matters, PMNS = makeProbs(energies, 4, ordering)
    values.append(a[0])

values=np.asarray(values)

for i, val in enumerate(values):
    if np.abs(val-probVacuum) < 0.005:
        print(permutations_array[i])

ax[1].plot(np.arange(len(permutations_array)), np.abs(values-probVacuum), linestyle='', marker='.', color='blue')
#ax[1].show()
print(PMNS2[1,1]-PMNS4[1,1])
print(PMNS4[0,3])
"""
