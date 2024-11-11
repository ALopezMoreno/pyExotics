import sys

sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
import multiprocessing
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from graphing import plotting
from mpl_toolkits.mplot3d import Axes3D


def solar_density(l):
    solar_radius = 696340
    density = 245 * np.exp(-10.54 * l / solar_radius)
    return density


nPoints = 50

density = np.logspace(0, 4, num=nPoints)
energies = np.logspace(-2 , 1, num=nPoints)

# SET PROPAGATORS FOR THE ADEQUATE NUMBER OF FLAVOURS
prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 10**3, 10 ** 0, False, False, 0, ngens=4,
                                              neOverNa=True)
# Set non-unitary parameters
S14 = 0.001 * 1
S24 = 0.2 * 1
S34 = 0.4 * 1
Hij = 0.0
sterileMass = 10 ** 4

prop.mixingPars = [np.arcsin(np.sqrt(0.307)),
                   np.arcsin(np.sqrt(0.022)),
                   np.arcsin(np.sqrt(0.561)),
                   np.arcsin(np.sqrt(S14)),
                   np.arcsin(np.sqrt(S34)),
                   np.arcsin(np.sqrt(S24)),
                   0, Hij, Hij]

# SET COMPLEX PHASE TO ZERO
# prop.mixingPars[-1] = 0
prop.masses.append(sterileMass)
prop.set_gens(4)


G_f = 5.4489e-5
U = np.real(prop.vHam)
#print(prop.vHam)
chi = U[1, 1] * U[2, 2] - U[1, 2] * U[2, 1]
print('determinant of A is:')
print(np.linalg.det(U))
print('Chi is:')
print(chi)

c11 = U[1:, 1:]
c44 = U[:3, :3]
b = 2/3
xi = np.linalg.det(c11) - np.linalg.det(c44)*b
print('Xi is:')
print(xi)
factor = 2 * G_f * np.sqrt(2)

lims = xi / (chi * b) / factor
#print('lim is:')
#print(lim)
probs = np.ones((nPoints, nPoints))

for k, rho in tqdm(enumerate(density), total=len(density)):
    for h, es in enumerate(energies):

        prop.update_hamiltonian(es, rho, ngens=4, neOverNa=True)
        probs[k, h] = prop.getOsc(0, 0)

fig, ax = plt.subplots(dpi=250)
plotting.plot2Dcontour(ax, density, energies, probs.T, logx=True, logy=True)
ax.set_box_aspect(1)
ax.set_xlabel(r"$n_e/N_a$")
ax.set_ylabel(r"$E_{\nu}$ (GeV)")
ax.scatter(lims / energies, energies, color='red', s=2)
plt.show()
