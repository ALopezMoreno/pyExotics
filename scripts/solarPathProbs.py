import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore
from tqdm import tqdm
from graphing import  plotting
# Solar electron number density
def solar_density(l):
    solar_radius = 696340
    density = 245*np.exp(-10.54*l/solar_radius)
    return density

def vacuum(l):
    return 0.0

# Plotting options
plotApprox = False
plotPvsE = True
getVacuum = False


energyMax = float(sys.argv[1])
energyMin = float(sys.argv[2])
energyBin = int(sys.argv[3])

# Make list of energies to loop through
energies = np.logspace(energyMin, energyMax, energyBin)


matterHam = customPropagator.matterHamiltonian
prop = customPropagator.HamiltonianPropagator(0, 1, 1)
ne_profile = solar_density


max_change = float(sys.argv[4])
savefile = float(sys.argv[5])

solver = customPropagator.VaryingPotentialSolver(prop, matterHam,
                                                 ne_profile, 0, 696340,
                                                 max_change)

print('nbins = ' + str(len(solver.binCentres)))
probs = np.zeros(energyBin)

# calculate transition amplitude from the surface of the sun to infinity
# this does not depend on energy
if getVacuum == True:
    solver.propagator.newHam = np.zeros((3, 3))
    solver.propagator.E = 10**-3
    solver.propagator.update()
    eigenvalues = solver.propagator.eigenvals
    lovere = 10 ** 15
    lengths = lovere / np.linspace(0.8, 1.2, 10 ** 4)
    vacuum_amps = np.zeros((3, 3), dtype=complex)

    for j in lengths:
        solver.propagator.L = j
        for k in range(3):
            for g in range(3):
                vacuum_amps[k, g] += solver.propagator.getAmps(k, g)

    vacuum_amp = vacuum_amps / len(lengths)
    print(vacuum_amp[0, 0])
    print(np.abs(vacuum_amp[0, 0] * vacuum_amp[0, 0].conjugate()))


for i, energy in tqdm(enumerate(energies*10**-3), total=len(energies)):
    solver.propagator.E = energy

    # calculate transition amplitude inside the sun
    solver.setTransitionAmplitude()
    probs[i] = solver.getProbs(0, 0)
    #solar_amps = solver.getTransitionAmplitude()
    #angles[i] = 0.308

    #total_amp = solar_amps[0, 0] #np.matmul(vacuum_amp.transpose(), solar_amps)[0, 0]
    #probs[i] = np.abs(total_amp * total_amp.conjugate())

data = np.column_stack((energies, probs))
np.savetxt(savefile, data, delimiter="\t", fmt='%.9f')




