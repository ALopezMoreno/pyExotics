import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../')

from HamiltonianSolver import customPropagator
from graphing import plotting


def setSterileMixing(propagator, scale, h4gen):
    # Set up a 4x4 customPropagator object from a default 3x3 one
    propagator.masses = [0, np.sqrt(7.42 * 10 ** (-5)), np.sqrt(2.51 * 10 ** (-3)), 10 ** 6]
    propagator.mixingPars = [np.arcsin(np.sqrt(0.307)), np.arcsin(np.sqrt(0.022)), np.arcsin(np.sqrt(0.561)),
                           np.arcsin(np.sqrt(scale)), np.arcsin(np.sqrt(scale)), np.arcsin(np.sqrt(scale)), -1.601, 0.0, 0.0]
    propagator.generations = 4
    propagator.newHam = h4gen
    propagator.update()

def matterHamiltonian(density, ngens):
    # set up nominal matter hamiltonian for n dimensions
    H = np.zeros((ngens, ngens))
    H[0, 0] = density * 1.663787e-5 * np.sqrt(2)
    if ngens > 3:
        for i in range(3, ngens):
            H[i, i] = -2*H[0, 0]
    return H

def getAvgProbs(alpha, beta, Vin, Vout):
    prob = 0
    for i in range(len(Vin)):
        prob += Vin[alpha, i] * Vout[beta, i].conjugate()
    P = np.abs(prob*prob.conjugate())
    return P

# Calculate the averaged out solar to earth probabilities.
n_solar = 100 #  electron density in the centre of the sun
n_earth = 0   #  electron density at a surface experiment

h_solar_3gen = matterHamiltonian(n_solar, 3)
h_earth_3gen = matterHamiltonian(n_earth, 3)

h_solar_4gen = matterHamiltonian(n_solar, 4)
h_earth_4gen = matterHamiltonian(n_earth, 4)

npoints = 200
E = np.logspace(0, 3, npoints)  #  Energy in MeV
nonU_scale = 0.00 #  np.linspace(0, 0.5, npoints)

prop3gen = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 10**10, 1*10**-2, False, False, 0, ngens=3)
prop4gen = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 10**10, 1*10**-3, False, False, 0, ngens=3)

prop3gen.update()
prop4gen.update()
# Get mixing matrices in nominal scenario
V3_solar = prop3gen.mixingMatrix
prop3gen.newHam = h_earth_3gen
prop3gen.update()
V3_earth = prop3gen.mixingMatrix

# Finally, calculate oscillation probs
P3_ee = getAvgProbs(0, 0, V3_solar, V3_earth)
P3_em = getAvgProbs(0, 1, V3_solar, V3_earth)
P3_et = getAvgProbs(0, 2, V3_solar, V3_earth)
P3 = np.array([P3_ee, P3_em, P3_et])

print(P3)

probs = np.zeros((npoints, 3))

setSterileMixing(prop4gen, nonU_scale, h_solar_4gen)
for i, scale in tqdm(enumerate(E)):
    # Get ,mixing matrix in sterile scenario
    prop4gen.E = scale*10**-3
    prop4gen.newHam = h_solar_4gen
    prop4gen.update()
    V4_solar = prop4gen.mixingMatrix
    prop4gen.newHam = np.zeros((4, 4))#h_earth_4gen
    prop4gen.update()
    V4_earth = prop4gen.mixingMatrix

    probs[i, 0] = getAvgProbs(0, 0, V4_solar, V4_earth)
    probs[i, 1] = getAvgProbs(0, 1, V4_solar, V4_earth)
    probs[i, 2] = getAvgProbs(0, 2, V4_solar, V4_earth)


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400)
plt.grid(True, which="both", axis='x', linestyle='--', linewidth=0.8)

#  Set colours and legends
colourses = ['r', 'b', 'gold']
legends = [r'$P_{ee}$', r'$P_{e\mu}$', r'$P_{e\tau}$']

ax.set_ylim(0, 1)
#ax.set_xlim(0, 0.5)

plt.title(r'Averaged adiabatic probabilities from solar neutrinos Non-Unitarity = ' + str(nonU_scale))
for i in range(3):
    ax.axhline(y=P3[i], color=colourses[i], linestyle='--', linewidth=1.5)
    plotting.niceLinPlot(ax, E, probs[:, i], logy=False, color=colourses[i], linewidth=1.5, label=legends[i])

#ax.set_xlabel(r'$\sin^2\theta_{i4}$', fontsize=15)
ax.set_xlabel(r'E (MeV)')
ax.set_ylabel(r'$P_{e \alpha}$', fontsize=15)
plt.legend(loc='upper left')
plt.savefig('../images/adiabaticSolarNonU_0.png')