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

def solar_density(l):
    solar_radius = 696340
    density = 245*np.exp(-10.54*l/solar_radius)
    return density


nPoints = 10 * int(1000/30)
#density = np.linspace(0.0001, 10, num=nPoints)
path = np.linspace(0, 696340, num=nPoints)
#density = solar_density(path)
density = np.logspace(-5, 5, num=nPoints)

# SET PROPAGATORS FOR THE ADEQUATE NUMBER OF FLAVOURS
prop = customPropagator.HamiltonianPropagator(0, 1, 1)

prop.masses = prop.masses[:-1]
print(prop.masses)
prop.generations = 2

prop.mixingPars = [np.arcsin(np.sqrt(0.32))]
prop.update()

# GET EIGEN-OBJECTS
matrices = np.ones((2, 2, nPoints))
eigenvals = np.ones((2, nPoints))
for k, rho in tqdm(enumerate(density), total=len(density)):

    matterH = customPropagator.matterHamiltonian(rho, ngens=2, neOverNa=True)
    prop.new_hamiltonian(matterH)
    P = prop.mixingMatrix
    D = prop.eigenvals
    matrices[:, :, k] = np.real(P)
    eigenvals[:, k] = np.real(D)
# FIX SIGNS
summed_vectors = np.sum(matrices, axis=1)

nonneg = summed_vectors >= 0

# create an integer array with values -1 or 1 depending on the sign of the coordinates
signs = np.ones(summed_vectors.shape[1], dtype=int)
signs[~nonneg.all(axis=0)] = -1

matrices = matrices * signs

# Multiply matrix array and integer array element-wise

# PLOTTING
# create a function that updates the plot for each time step
def update_plot(frame, vectors, densities, color='blue'):
    # clear the current plot
    plt.cla()

    # Plot flavour vectors:
    plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='red', label=r'Flavour states')
    plt.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='red')
    # plot all the vectors up to the current frame
    #for i, col in enumerate(vectors[0, :, 0]):
    for k, v in enumerate(vectors[:, :, frame+1]):
        if k == 0:
            plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=color, label=r'Mass states')
        else:
            plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=color)


    # set the x and y limits of the plot
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)

    # set the title of the plot and the axes
    plt.title(f"$n_e/N_a$ = {densities[frame+1]:.4f}")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_box_aspect(1)
    legend = ax.legend(loc='lower left')
    # legend.get_texts()[-1].remove()# create a figure and axis object
fig, ax = plt.subplots(dpi = 200)


# create an animation object that updates the plot for each frame
anim = FuncAnimation(fig, update_plot, fargs=(matrices, density,), frames=tqdm(range(nPoints-1)), interval=1000/30)
#plotting.niceLinPlot(ax, density, eigenvals[0, :], logx=True, color='blue', linestyle='', marker='o', markersize=1)
#plotting.niceLinPlot(ax, density, eigenvals[1, :], logx=True, color='red', linestyle='', marker='o', markersize=1)

#plt.show()
# save the animation as a GIF file
anim.save('../images/2flavour_densityEvolution.gif', writer='pillow')