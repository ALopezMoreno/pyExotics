import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from graphing import plotting

matrices = np.load('../eigenvectors/eigs.bin.npy')
th12 = 0.5872523687443223
th13 = 0.14887328003763659
th23 = 0.8465505066509045

ve_path = matrices[0, :, :]
vmu_path = matrices[1, :, :]
vtau_path = matrices[2, :, :]

base_theta = np.arccos(ve_path[2, :])*-1 + np.pi / 2 # inclination (latitude)
base_phi = np.arctan2(ve_path[1, :], ve_path[0, :])  # azimuth (longitude)
theta= 1/np.sin(0.5*base_theta)
phi = np.tan(base_phi)

fig, ax = plt.subplots(dpi=200)
col = 'gold'
color_values = range(len(phi))
plotting.niceLinPlot(ax, phi, theta, logx=False, logy=False, linestyle='', marker='o',
                     markersize=1, color=col)
ax.plot(phi[0], theta[0], color=col, marker='o', markersize=4)

""""
col = 'lightskyblue'
theta = np.arccos(vmu_path[2, :])*-1 + np.pi / 2 # inclination (latitude)
phi = np.arctan2(vmu_path[1, :], vmu_path[0, :])  # azimuth (longitude)
plotting.niceLinPlot(ax, np.sin(phi), theta, logx=False, logy=False, linestyle='', marker='o',
                     markersize=1, color=col)
ax.plot(phi[0], theta[0], color=col, marker='o', markersize=4)

col = 'plum'
theta = np.arccos(vtau_path[2, :])*-1 + np.pi / 2 # inclination (latitude)
phi = np.arctan2(vtau_path[1, :], vtau_path[0, :])  # azimuth (longitude)
plotting.niceLinPlot(ax, np.sin(phi), theta, logx=False, logy=False, linestyle='', marker='o',
                     markersize=1, color=col)
ax.plot(np.sin(phi[0]), theta[0], color=col, marker='o', markersize=4)


ax.axvline(x=np.sin(th12), color='red', linewidth=1)
ax.axhline(y=th13, color='limegreen', linewidth=1)
#ax.axvline(x=th23+th12, color='blue', linewidth=1)
"""

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\theta$")
#ax.set_xlim(-1, 1)
#ax.set_ylim(-np.pi/2, np.pi/2)

#ax.hlines([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], -np.pi, np.pi, 'k', linestyle='--', linewidth=0.5)  # Draw horizontal lines
#ax.vlines(np.sin([-np.pi, -np.pi/2, 0, np.pi/2, np.pi]), -np.pi/2, np.pi/2, 'k', linestyle='--', linewidth=0.5)  # Draw vertical lines
ax.set_box_aspect(1)

plt.title(r"Path of $\nu_e$ through $n_e$")
plt.savefig('../images/testPath.png')
plt.show()
