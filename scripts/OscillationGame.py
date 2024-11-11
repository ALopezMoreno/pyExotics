import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
from tqdm import tqdm
import matplotlib.pyplot as plt
from graphing import plotting
import matplotlib.ticker as ticker

# THIS SCRIPT PLOTS OSCILLATION PROBS FOR 3 VS 4 NEUTRINOS

npoints = 10**3
E = 2
rho = 2.7/2
lengths = np.logspace(-1, 3, npoints)

sterileMass = 100 # eV^2

S14 = 0.2
S24 = 0.2
S34 = 0.2
Hij = 0

probs_3nu = np.empty(npoints)
probs_4nu = np.empty(npoints)


prop_3nu = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 1, E,
                                               False,
                                               False,  0, ngens=3)

prop_4nu = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 1, E,
                                               False,
                                               False,  0, ngens=4)


prop_4nu.masses.append(sterileMass)
prop_4nu.mixingPars = [np.arcsin(np.sqrt(0.307)),
                   np.arcsin(np.sqrt(0.022)),
                   np.arcsin(np.sqrt(0.561)),
                   np.arcsin(np.sqrt(S14)),
                   np.arcsin(np.sqrt(S24)),
                   np.arcsin(np.sqrt(S34)),
                   -1.601, Hij, Hij]

prop_4nu.set_gens(4)

prop_3nu.update_hamiltonian(E, rho, ngens=3)
prop_4nu.update_hamiltonian(E, rho, ngens=4)

for i, L in tqdm(enumerate(lengths)):

    prop_3nu.L = L
    prop_4nu.L = L

    prop_3nu.update()
    prop_4nu.update()

    probs_3nu[i] = prop_3nu.getOsc(1, 1)
    probs_4nu[i] = prop_4nu.getOsc(1, 1)



# Plotting
fig, ax = plt.subplots(dpi=200)

plotting.niceLinPlot(ax, lengths, probs_3nu, logy=False, color='blue')
plotting.niceLinPlot(ax, lengths, probs_4nu, logy=False, color='gold', alpha= 0.7)
ax.axvline(x=297, color='red', linestyle='dashed')
ax.axvline(x=0.280, color='red', linestyle='dashed')
plt.show()