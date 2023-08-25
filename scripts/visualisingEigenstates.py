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


nPoints = 10 * int(1000 / 30)
# density = np.linspace(0, 100, num=nPoints)
# path = np.linspace(0, 696340, num=nPoints)
# density = solar_density(path)
density = np.logspace(-1, 2, num=nPoints)
# density = np.linspace(3*10**2, 5*10**2, num=nPoints)
# SET PROPAGATORS FOR THE ADEQUATE NUMBER OF FLAVOURS
prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, 10**3, 10 ** 0, False, False, 0, ngens=4,
                                              neOverNa=True)
# Set non-unitary parameters
S14 = 0.01 * 1
S24 = 0.01 * 1
S34 = 0.05 * 1
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
# print(prop.PMNS)

G_f = 5.4489e-5

prop.E = 10 ** 0
# print(prop.vHam)
# GET EIGEN-OBJECTS
matrices = np.ones((4, 4, nPoints))
matrices2 = np.ones((4, 4, nPoints))
sum_bottom = np.ones(nPoints)
eigenvals = np.ones((4, nPoints))
probs = np.ones(nPoints)
old = np.zeros((4, 4))

# print(prop.masses[1]*np.cos(2*prop.mixingPars[0])/(2*np.sqrt(2)*prop.E*5.4489e-5))
# print(prop.masses[1]**2)

for k, rho in tqdm(enumerate(density), total=len(density)):
    if k >= 1:
        old = matrices[:, :, k - 1]
    # matterH = customPropagator.matterHamiltonian(10**-3, rho, ngens=3, neOverNa=True)
    prop.update_hamiltonian(10 ** 0, rho, ngens=4, neOverNa=True)
    # print(prop.newHam)
    P = prop.mixingMatrix #.conjugate()  # np.linalg.inv(prop.mixingMatrix)
    D = prop.eigenvals
    matrices[:, :, k] = P
    matrices2[:, :, k] = prop.hamiltonian
    eigenvals[:, k] = np.abs(D)
    sum_bottom[k] = np.sum(matrices2[:, -1, k])
    probs[k] = prop.getOsc(0, 0)
    # if np.min(np.abs(D)) < 10**-7:
    # print()
    # print(np.real(P))
    # print(np.real(prop.vHam))
    # print(prop.newHam)

U = np.real(prop.vHam)
#print(prop.vHam)
#print(prop.PMNS)
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
factor = 2 * 10**0 * G_f * np.sqrt(2)

#lim = (xi + np.sqrt(xi ** 2 + 4 * chi)) / (2 * chi) #/ factor
lim = xi / (chi * b) / factor
print('lim is:')
print(lim)
# np.save('../eigenvectors/eigs.bin', matrices)
# FIX SIGNS
# summed_vectors = np.sum(matrices, axis=1)
# print(prop.hamiltonian)
# print(matrices[:, :, -1])
# nonneg = summed_vectors >= 0

# create an integer array with values -1 or 1 depending on the sign of the coordinates
# signs = np.ones(summed_vectors.shape[1], dtype=int)
# signs[~nonneg.all(axis=0)] = -1

# matrices = matrices * signs

# Multiply matrix array and integer array element-wise
print('done!')


# PLOTTING
# create a function that updates the plot for each time step
def update_plot(frame, ax, vectors, vectors2, densities, color='blue'):
    # clear the current plot
    global trace, trace2
    plt.cla()

    # Plot flavour vectors:
    basecol = 'red'
    ax.quiver(-1, 0, 0, 1, 0, 0, length=2, color=basecol, arrow_length_ratio=0.1, label=r'Mass states')
    ax.quiver(0, -1, 0, 0, 1, 0, length=2, color=basecol, arrow_length_ratio=0.1)
    ax.quiver(0, 0, -1, 0, 0, 1, length=2, color=basecol, arrow_length_ratio=0.1)
    ax.text(1.25, 0, 0, r'$\nu_{1}$', fontsize=10, color=basecol, ha='center', fontweight='bold')
    ax.text(0, 1.25, 0, r'$\nu_{2}$', fontsize=10, color=basecol, ha='center', fontweight='bold')
    ax.text(0, 0, 1.25, r'$\nu_{3}$', fontsize=10, color=basecol, ha='center', fontweight='bold')

    # plt.quiver(0, 0, 0, 0, -1, 0, length=1, color='red')
    # plt.quiver(0, 0, 0, 0, 0, -1, length=1, color='red')

    # plot all the vectors up to the current frame
    # for i, col in enumerate(vectors[0, :, 0]):
    myDict = {0: r"e", 1: r"\nu", 2: r"\tau", 3: r"\xi"}
    for k, v in enumerate(vectors[:, :, frame + 1]):
        name = myDict[k]

        if k == 0:
            for p, component in enumerate(v):
                trace[p].append(component)
                # trace2[p].append(vectors2[k, p, frame + 1])

            ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1, normalize=True, color=color, arrow_length_ratio=0.2,
                      label=r'Flavour states')
            ax.text(v[0] * 1.15, v[1] * 1.15, v[2] * 1.15, fr'$\nu_{name}$', fontsize=10, color=color, ha='center')
            ax.plot(trace[0], trace[1], trace[2], marker='o', color='gold', markersize=1, linestyle='')
            # ax.plot(trace2[0], trace2[1], trace2[2], marker='o', color='orange', markersize=1, linestyle='')

            # plt.quiver(0, 0, 0, -v[0], -v[1], -v[2], length=1, normalize=True, color=color)
        elif k < 3:
            ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1, normalize=True, color=color, arrow_length_ratio=0.2)
            ax.text(v[0] * 1.15, v[1] * 1.15, v[2] * 1.15, fr'$\nu_{name}$', fontsize=10, color=color, ha='center')
        else:
            ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1, normalize=True, color='grey', arrow_length_ratio=0.1,
                      linewidth=0.5)
            ax.text(v[0] * 1.15, v[1] * 1.15, v[2] * 1.15, fr'$\nu_{name}$', fontsize=10, color='grey', ha='center')

    # set the x and y limits of the plot
    # plt.xlim(-1, 5)
    # plt.ylim(-1, 5)

    # set the title of the plot and the axes
    plt.title(f"$n_e/N_a$ = {densities[frame + 1]:.4f}")
    # plt.suptitle(r'Evolution of eigenstates for LMA solution and $\sin^2\theta$ = 0.32')

    ax.set_xlim3d(-1.1, 1.1)
    ax.set_ylim3d(-1.1, 1.1)
    ax.set_zlim3d(-1.1, 1.1)

    # ax.set_box_aspect(1)
    legend = ax.legend(loc='lower left')
    # legend.get_texts()[-1].remove()# create a figure and axis object


#density = density
fig, ax = plt.subplots(dpi=250)
plotting.niceLinPlot(ax, density, eigenvals[0, :], logx=True, logy=True, color='blue', linestyle='', marker='o',
                     markersize=1)
plotting.niceLinPlot(ax, density, eigenvals[1, :], logx=True, logy=True, color='red', linestyle='', marker='o',
                     markersize=1)
plotting.niceLinPlot(ax, density, eigenvals[2, :], logx=True, logy=True, color='limegreen', linestyle='', marker='o',
                     markersize=1)
plotting.niceLinPlot(ax, density, eigenvals[3, :] * 10 ** 0, logx=True, logy=True, color='gold', linestyle='',
                     marker='o',
                     markersize=1)

ax.axvline(x=lim, color='black', linewidth=1)
# plotting.niceLinPlot(ax, density, sum_bottom, logx=True, logy=True, color='black', linestyle='',
#                     marker='o',
#                     markersize=1)

"""
plotting.niceLinPlot(ax, density, eigenvals2[0, :], logx=True, logy=True, color='cyan', linestyle='', marker='o',
                     markersize=1)
plotting.niceLinPlot(ax, density, eigenvals2[1, :], logx=True, logy=True, color='magenta', linestyle='', marker='o',
                     markersize=1)
plotting.niceLinPlot(ax, density, eigenvals2[2, :], logx=True, logy=True, color='limegreen', linestyle='', marker='o',
                     markersize=1)
"""

plt.title(fr'Evolution of eigenvalues in matter for $E_\nu={prop.E * 10 ** 3}MeV$')
ax.set_xlabel(r"$n_e/N_a$")
ax.set_ylabel(r"$\hat{m^2}$")
plt.savefig('../images/4flavour_densityEvolution.png')
plt.show()

"""
indx = np.abs(density - 245).argmin()

theta = np.arccos(matrices[0, 2, indx])
energies = np.logspace(-4, -1, 200)  #this corresponds to 0.1-100 MeV
phi = np.arctan(matrices[0, 1, indx] / matrices[0, 0, indx])
P_ee = 1 - (np.sin(2*phi)**2 * np.sin(10 * 7 * 10**-5 / (2*energies))**2 + 1/2*np.cos(theta)**2)
plt.cla()
plotting.niceLinPlot(ax, energies, P_ee, logy=False, color='gold')
plt.plot()
# plotting.niceLinPlot(ax, density, eigenvals[1, :]-eigenvals[0, :], logx=True, logy =True, color='goldenrod', linestyle='--', marker=None, markersize=1)
plt.show()
"""
fig, ax = plt.subplots(dpi=250)
plotting.niceLinPlot(ax, density, probs, logx=True, logy=False, color='blue', linestyle='', marker='o', markersize=1, alpha=0.5)
ax.axvline(x=lim, color='black', linewidth=1)
plt.show()


#fig = plt.figure(dpi=200)
#ax = fig.add_subplot(111, projection='3d')
#ax.view_init(elev=30, azim=45)
#trace2 = {0: [], 1: [], 2: [], 3: []}
#trace = {0: [], 1: [], 2: [], 3: []}
# create an animation object that updates the plot for each frame
#anim = FuncAnimation(fig, update_plot, fargs=(ax, matrices, matrices2, density,), frames=tqdm(range(nPoints - 1)),
#                    interval=1000 / 30)

resonanceIndex = np.argmin(np.abs(np.log(eigenvals[1, :]) - np.log(eigenvals[0, :])))
# plt.axvline(x=density[resonanceIndex], color='black')
print(matrices[:, :, -1])
print()
print(matrices[:, :, resonanceIndex])
print()
print(np.real(prop.PMNS))


# plt.show()
# save the animation as a GIF file
#anim.save('../images/4flavour_densityEvolutionRes.gif', writer='pillow')
