import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from graphing import plotting
from itertools import permutations

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

energyMin = np.log10(0.03)
energyMax = np.log10(11)
nEnergies = 1000
nGens = 4

S14 = 0.05
S24 = 0.01
S34 = 0.05
Hij = 0

th14 = np.arcsin(np.sqrt(S14))
th24 = np.arcsin(np.sqrt(S24))
th34 = np.arcsin(np.sqrt(S34))
print(np.cos(th14))
print(np.sin(th14))
# Make list of energies to loop through
energies = np.logspace(energyMin, energyMax, nEnergies)

def makeProbs(energies, nGens):
    # Set up propagator
    matterHam = customPropagator.matterHamiltonian

    prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 1, 1, False, False, 0,
                                                  ngens=nGens)
    prop.generations = nGens
    global S14, S24, S34, Hij
    if nGens == 4:
        sterileMass = 10 ** 4
        # """
        prop.masses.append(sterileMass)

        prop.mixingPars = [np.arcsin(np.sqrt(0.307)),
                           np.arcsin(np.sqrt(0.022)),
                           np.arcsin(np.sqrt(0.561)),
                           np.arcsin(np.sqrt(S14)),
                           np.arcsin(np.sqrt(S24)),
                           np.arcsin(np.sqrt(S34)),
                           -1.601, Hij, Hij]

        prop.update()
        # """
    a = []
    matters = []
    for i, e in tqdm(enumerate(energies), total=nEnergies):
        prop.update_hamiltonian(e * 10 ** -3, 245, ngens=nGens, neOverNa=True)
        prop.update()
        mixingMatrix = prop.mixingMatrix
        vacuumMatrix = prop.PMNS
        matt = prop.newHam[0, 0]
        P_ee = 0
        for k in range(3):
            P_ee += np.absolute(mixingMatrix[0, k].conjugate() * vacuumMatrix[0, k]) ** 2

        a.append(P_ee)
        matters.append(matt)
    return a, matters, prop.PMNS


a, matters, PMNS4 = makeProbs(energies, 4)
a0, matters0, PMNS = makeProbs(energies, 3)


fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Set logarithmic scales for x and y axes
plt.xscale('log')


# Plot the data with a blue line
plt.plot(energies, a, color='blue', alpha=0.6, linewidth=2, label='3+1H (numerical solution)')
plt.plot(energies, a0, color='red', alpha=0.6, linewidth=2, label=r'3$\nu$ (numerical solution)')

# Set tick positions and labels
#plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('bottom')
plt.gca().yaxis.tick_left()
plt.gca().yaxis.set_label_position('left')
plt.minorticks_on()
plt.gca().xaxis.set_tick_params(which='both', direction='in', width=1, length=5)
plt.gca().yaxis.set_tick_params(which='both', direction='in', width=1, length=5)
plt.gca().xaxis.set_tick_params(which='minor', direction='in', width=1, length=2)
plt.gca().yaxis.set_tick_params(which='minor', direction='in', width=1, length=2)
# Increase the size of tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.ylim(0,0.6)

# Set axis labels
plt.xlabel(r'E (MeV)', fontsize=15)
plt.ylabel(r'$P_{ee}$', fontsize=15)

# Add a grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)




th12 = np.arcsin(np.sqrt(0.308))
th13 = np.arcsin(np.sqrt(0.022))
lma_condition = np.cos(2 * th12)
msw_condition = 1

dm12 = (7.42 * 10 ** (-5))
vacuum_2f = 1 - 0.5 * np.sin(2 * th12) ** 2

vac_prob = np.cos(th13) ** 4 * vacuum_2f + np.sin(th13) ** 4
msw_prob = np.sin(th12) ** 2 * np.cos(th13) ** 4


beta = (2 * np.sqrt(2) * 5.3948e-5 * np.cos(th13) ** 2 * 245 * energies * 10 ** -3) / (7.42 * 10 ** (-5))
matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + np.sin(2 * th12) ** 2)
probsLMA = np.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) #+ np.sin(th13)**4



alpha = np.asarray([[np.cos(th14), 0, 0],
                    [-np.sin(th14)*np.sin(th24), np.cos(th24), 0],
                    [-np.sin(th34)*np.cos(th24)*np.sin(th14), -np.sin(th24)*np.sin(th34), np.cos(th34)]])

PMNS2 = np.matmul(alpha, PMNS)

print(np.linalg.det(alpha))
print(np.cos(th14)**2)

probVacuum = np.cos(th14)**4 - 4*(np.abs(PMNS2[0, 0])**2*np.abs(PMNS2[0,1])**2/2 +
                                  np.abs(PMNS2[0, 0])**2*np.abs(PMNS2[0,2])**2/2 +
                                  np.abs(PMNS2[0, 1])**2*np.abs(PMNS2[0,2])**2/2)

beta = beta * np.cos(th14)**4 #/ np.cos(th24)**2 #- 4*np.sin(th14)*np.sin(th24)*np.cos(th24) *np.sin(th12)*np.cos(th12)
matterAngle = (np.cos(2 * th12) - beta) / np.sqrt((np.cos(2 * th12) - beta) ** 2 + (np.sin(2 * th12)) ** 2)
probsLMA2 = np.cos(th13) ** 4 * (1 / 2 + 1 / 2 * matterAngle * np.cos(2 * th12)) + np.sin(th13)**4
probsLMA2 *= (1-S14)**2


#closest_index2 = np.argmin(np.abs(matters - crit2))
#closest_value2 = energies[closest_index2]

plt.plot(energies, probsLMA, color='red', linewidth=2, linestyle='-.', label=r'3$\nu$ LMA approximation')
plt.plot(energies, probsLMA2, color='blue', linewidth=2, linestyle='-.', label='3+1H LMA approximation')
plt.ylim(0,0.8)


#plt.axhline(y=probVacuum, color='blue')
#plt.axhline(y=msw_prob, color='red', xmin=closest_index/nEnergies)

#plt.axhline(y=vac_prob * (1-S14)**2, color='green', xmax=closest_index/nEnergies)
#plt.axhline(y=msw_prob - 1+(1-S14), color='red', xmin=closest_index/nEnergies)

#plt.axvline(x=closest_value)
plt.legend()
plt.title('Electron survival probabilities in matter - decoherent mixing')
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

plt.plot(np.arange(len(permutations_array)), np.abs(values-probVacuum), linestyle='', marker='.', color='blue')
#plt.show()
print(PMNS2[1,1]-PMNS4[1,1])
print(PMNS4[0,3])
"""
