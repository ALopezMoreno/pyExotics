import sys

import matplotlib.pyplot as plt

sys.path.append("../")
import numpy as np
from graphing import plotting


# Open the text file for reading
with open("solarFluxes/BorexinoKamland.txt", "r") as file:
    lines = file.readlines()

# Initialize empty lists to store the data
be7_fluxes = []
b8_fluxes = []
reduced_chi2 = []

# Loop through the lines and extract data
for line in lines[3:]:  # Skip the header lines
    values = line.split()

    be7_fluxes.append(float(values[0]))
    b8_fluxes.append(float(values[1]))
    reduced_chi2.append(float(values[2]))

be7_fluxes = np.asarray(be7_fluxes)
b8_fluxes = np.asarray(b8_fluxes)
reduced_chi2 = np.asarray(reduced_chi2)

print(np.min(reduced_chi2))
# Reshape into readable form for plot2Dcontour:
y = np.unique(be7_fluxes)
x = np.unique(b8_fluxes)

num_y_values = len(y)
num_x_values = len(x)
z_2d = reduced_chi2.reshape(num_x_values, num_y_values)

# Calculate percentile corresponding to 1 sigma
per = [100 * np.percentile(0.5*reduced_chi2, 1)]

# Get best fit point:
besty, bestx = np.unravel_index(np.argmin(z_2d.T), z_2d.T.shape)

# Plot
fig, ax = plt.subplots(dpi=300)
plotting.plot2Dcontour(ax, x, y, z_2d.T,
                       nbins=10,
                       reverse=True,
                       cmap=False,
                       fill=True,
                       col=('black',),
                       percentile=np.asarray(per))


ax.set_ylim(3.8*10**9, 5.8*10**9)
ax.set_xlim(2.5*10**6, 7.5*10**6)
plotting.makeTicks(ax, allsides=True, ynumber=4)
ax.scatter(x[bestx], y[besty], marker='o', color='black', s=7)
ax.set_ylabel(r'$\phi_{BE}$ $[\nu$ cm$^{-2}$s$^{-1}]$')
ax.set_xlabel(r'$\phi_{B}$ $[\nu$ cm$^{-2}$s$^{-1}]$')

ax.grid(linestyle='dashed')
ax.set_box_aspect(1)
plt.show()