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
    density = 245*np.exp(-10.54*l/solar_radius)
    return density


nPoints = 5 * int(1000/30)
# density = np.linspace(0, 100, num=nPoints)
path = np.linspace(0, 696340, num=nPoints)
density = solar_density(path)
# density = np.logspace(0, 6, num=nPoints)
#density = np.linspace(0, 10**3, num=nPoints)
# SET PROPAGATORS FOR THE ADEQUATE NUMBER OF FLAVOURS
prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 1, 10-3, False, False, 0, ngens=3, neOverNa=True)
# print(prop.masses)
#prop.masses = prop.masses[:-1]
#prop.masses = [0, np.sqrt(7 * 10 ** -6)]

# prop.mixingPars = [np.arcsin(np.sqrt(0.32))] #0.32

# SET COMPLEX PHASE TO ZERO
prop.mixingPars[-1] = 0
print(prop.mixingPars)
prop.set_gens(3)

prop.E = 10 ** -2


# GET EIGEN-OBJECTS
matrices = np.ones((3, 3, nPoints))
eigenvals = np.ones((3, nPoints))
old = np.zeros((3, 3))
#print(prop.masses[1]*np.cos(2*prop.mixingPars[0])/(2*np.sqrt(2)*prop.E*5.4489e-5))
#print(prop.masses[1]**2)

for k, rho in tqdm(enumerate(density), total=len(density)):
    if k >= 1:
        old = matrices[:, :, k-1]
    matterH = customPropagator.matterHamiltonian(10**-2, rho, ngens=3, neOverNa=True)
    #prop.update_hamiltonian(rho, ngens=2, neOverNa=True)
    prop.newHam = matterH
    prop.update()

    P = prop.mixingMatrix
    D = prop.eigenvals
    matrices[:, :, k] = np.real(P)
    eigenvals[:, k] = np.real(D)

    #print(P)

# FIX SIGNS
# summed_vectors = np.sum(matrices, axis=1)
# print(prop.hamiltonian)
# print(matrices[:, :, -1])
# nonneg = summed_vectors >= 0

# create an integer array with values -1 or 1 depending on the sign of the coordinates
#signs = np.ones(summed_vectors.shape[1], dtype=int)
#signs[~nonneg.all(axis=0)] = -1

#matrices = matrices * signs

# Multiply matrix array and integer array element-wise

# PLOTTING
# create a function that updates the plot for each time step
def update_plot(frame, ax, vectors, densities, color='blue'):
    # clear the current plot
    plt.cla()

    # Plot flavour vectors:
    ax.quiver(-1, 0, 0, 1, 0, 0, length=2, color='red', arrow_length_ratio=0.1, label=r'Flavour states')
    ax.quiver(0, -1, 0, 0, 1, 0, length=2, color='red', arrow_length_ratio=0.1)
    ax.quiver(0, 0, -1, 0, 0, 1, length=2, color='red', arrow_length_ratio=0.1)
    ax.text(1.25, 0, 0, r'$\nu_{e}$', fontsize=10, color='red', ha='center', fontweight='bold')
    ax.text(0, 1.25, 0, r'$\nu_{\mu}$', fontsize=10, color='red', ha='center', fontweight='bold')
    ax.text(0, 0, 1.25, r'$\nu_{\tau}$', fontsize=10, color='red', ha='center', fontweight='bold')

    #plt.quiver(0, 0, 0, 0, -1, 0, length=1, color='red')
    #plt.quiver(0, 0, 0, 0, 0, -1, length=1, color='red')

    # plot all the vectors up to the current frame
    #for i, col in enumerate(vectors[0, :, 0]):
    for k, v in enumerate(vectors[:, :, frame+1]):
        name = k + 1
        if k == 0:

            ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1, normalize=True, color=color, arrow_length_ratio=0.2, label=r'Mass states')
            ax.text(v[0]*1.15, v[1]*1.15, v[2]*1.15, fr'$\nu_{name}$', fontsize=10, color=color, ha='center')

            # plt.quiver(0, 0, 0, -v[0], -v[1], -v[2], length=1, normalize=True, color=color)
        else:
            ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1, normalize=True, color=color, arrow_length_ratio=0.2)
            ax.text(v[0] * 1.15, v[1] * 1.15, v[2] * 1.15, fr'$\nu_{name}$', fontsize=10, color=color, ha='center')
            #plt.quiver(0, 0, 0, -v[0], -v[1], -v[2], length=1, normalize=True, color=color)


    # set the x and y limits of the plot
    #plt.xlim(-1, 5)
    #plt.ylim(-1, 5)

    # set the title of the plot and the axes
    plt.title(f"$n_e/N_a$ = {densities[frame+1]:.4f}")
    #plt.suptitle(r'Evolution of eigenstates for LMA solution and $\sin^2\theta$ = 0.32')

    ax.set_xlim3d(-1.1, 1.1)
    ax.set_ylim3d(-1.1, 1.1)
    ax.set_zlim3d(-1.1, 1.1)

    #ax.set_box_aspect(1)
    legend = ax.legend(loc='lower left')
    # legend.get_texts()[-1].remove()# create a figure and axis object

fig, ax =plt.subplots(dpi=250)
plotting.niceLinPlot(ax, density, eigenvals[0, :], logx=True, logy =True, color='blue', linestyle='', marker='o', markersize=1)
plotting.niceLinPlot(ax, density, eigenvals[1, :], logx=True, logy =True, color='red', linestyle='', marker='o', markersize=1)
plotting.niceLinPlot(ax, density, eigenvals[2, :], logx=True, logy =True, color='limegreen', linestyle='', marker='o', markersize=1)
plt.title(r'Evolution of eigenvalues in matter for $E_\nu=10MeV$')
ax.set_xlabel(r"$n_e/N_a$")
ax.set_ylabel(r"$\hat{m^2}$")
plt.savefig('../images/3flavour_densityEvolutionSolar.png')
#plotting.niceLinPlot(ax, density, eigenvals[1, :]-eigenvals[0, :], logx=True, logy =True, color='goldenrod', linestyle='--', marker=None, markersize=1)
plt.show()


fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=45)

# create an animation object that updates the plot for each frame
anim = FuncAnimation(fig, update_plot, fargs=(ax, matrices, density,), frames=tqdm(range(nPoints-1)), interval=1000/30)

#resonanceIndex = np.argmin(np.abs(np.log(eigenvals[1, :]) - np.log(eigenvals[0, :])))
#plt.axvline(x=density[resonanceIndex], color='black')
#print(density[resonanceIndex])


#plt.show()
# save the animation as a GIF file
anim.save('../images/3flavour_densityEvolutionSolar.gif', writer='pillow')