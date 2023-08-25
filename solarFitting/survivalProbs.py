import numpy as np
import sys
sys.path.append("../")
from graphing import plotting
from HamiltonianSolver import customPropagator
from tqdm import tqdm
import matplotlib.pyplot as plt
from solarFluxes import solarProbs as spr
import LMAmodels
from scipy.interpolate import interp1d

# LOAD BOREXINO DATA
pp = spr.pp
Be7 = spr.Be7
pep = spr.pep
B8 = spr.B8
B8_SNO = spr.B8_SNO


energyMin = np.log10(0.1)
energyMax = np.log10(20)
nEnergies = 1000
nGens = 4

th12 = np.arcsin(np.sqrt(0.307)) #0.308
th13 = np.arcsin(np.sqrt(0.022))
S14 = 0.05
S24 = 0.01
S34 = 0.05
Hij = 0

th14 = np.arcsin(np.sqrt(S14))
th24 = np.arcsin(np.sqrt(S24))
th34 = np.arcsin(np.sqrt(S34))

# Make list of energies to loop through
energies = np.logspace(energyMin, energyMax, nEnergies)

# Calculate LMA solutions
dm12 = (7.51 * 10 ** (-5))
probsLMA2 = LMAmodels.LMA_solution(energies, dm12, 0.36, 0.022)
probsLMA = LMAmodels.LMA_solution(energies, dm12, 0.307, 0.022)

# Plot LMA solutions
fig, ax = plt.subplots(dpi=300)

# Get fit results
sin2th12, sin2th13, dm21 = np.loadtxt('fitResults/out.txt').T

datas = np.empty((len(energies), len(dm21)))
for i in range(len(sin2th12)):
    datas[:, i] = LMAmodels.LMA_solution(energies, dm21[i],sin2th12[i], sin2th13[i])
    #plotting.niceLinPlot(ax, energies, datas[:, i], logy=False, color='r', alpha=0.05, linewidth=1)

# Calculate mean and standard deviation for each row
mean_per_row = np.mean(datas, axis=1)
std_per_row = np.std(datas, axis=1)

plt.fill_between(energies, mean_per_row - std_per_row, mean_per_row + std_per_row, color='limegreen', alpha=0.2)

ax.set_ylim(0,0.8)
plotting.makeTicks(ax, xdata=energies)
ax.set_ylabel(r'P$_{ee}$', fontsize=12)
ax.set_xlabel(r'E (MeV)', fontsize=12)


# Plot survival probs
colors = ['orange', 'green', 'red', 'purple', 'blue']
plt.errorbar(pp["Energy [MeV]"], pp["Pee"], yerr=[pp["Err Down"], pp["Err Up"]], fmt='.', capsize=0, ecolor=colors[0], color=colors[0])
plt.errorbar(Be7["Energy [MeV]"], Be7["Pee"], yerr=[Be7["Err Down"], Be7["Err Up"]], fmt='.', capsize=0, ecolor=colors[1], color=colors[1])
plt.errorbar(pep["Energy [MeV]"], pep["Pee"], yerr=[pep["Err Down"], pep["Err Up"]], fmt='.', capsize=0, ecolor=colors[2], color=colors[2])
plt.errorbar(B8["Energy [MeV]"], B8["Pee"], yerr=[B8["Err Down"], B8["Err Up"]], fmt='.', capsize=0, ecolor=colors[3], color=colors[3])
plt.errorbar(B8["Energy [MeV]"], B8["Pee"], yerr=[B8["Err Down"], B8["Err Up"]], fmt='.', capsize=0, ecolor=colors[3], color=colors[3])

# SNO data we treat slightly different
x_values = np.asarray(B8_SNO["Energy [MeV]"])
y_values = np.asarray(B8_SNO["Pee"])
y_err_down = np.asarray(B8_SNO["Err Down"])
y_err_up = np.asarray(B8_SNO["Err Up"])
# Create an interpolation function for the upper and lower bounds
interp_upper = interp1d(x_values, y_values + y_err_up, kind='cubic')
interp_lower = interp1d(x_values, y_values - y_err_down, kind='cubic')

# Generate points for smoother filling
x_smooth = np.linspace(min(x_values), max(x_values), 10)
y_upper_smooth = interp_upper(x_smooth)
y_lower_smooth = interp_lower(x_smooth)
plt.fill_between(x_smooth, y_lower_smooth, y_upper_smooth, color=colors[4], alpha=0.3)
#plt.errorbar(B8_SNO["Energy [MeV]"], B8_SNO["Pee"], yerr=[B8_SNO["Err Down"], B8_SNO["Err Up"]], fmt='.', capsize=0, ecolor=colors[4], color=colors[4])

# Calculate the position for the labels
vars = [pp, Be7, pep, B8, B8_SNO]
labels = [r'pp', r'$^7$Be', r'pep', r'$^8$B (Borexino)', r'$^8$B (SNO)']
for i, var in enumerate(vars):
    label_x = var["Energy [MeV]"][0]
    if i < 4:
        label_y = var["Pee"][0] + var["Err Up"][0] + 0.01  # Adjust the value to control label position
    else:
        label_y = var["Pee"][0] - var["Err Up"][0] - 0.1  # Adjust the value to control label position
    
    plt.annotate(labels[i],
                 xy=(label_x, label_y),
                 xytext=(5, 10), textcoords="offset points",
                 ha="center", fontsize=13, color=colors[i], weight='bold')

plotting.niceLinPlot(ax, energies, probsLMA, logy=False, color='b', label=r'LMA solution - NuFit', linestyle='dashed', linewidth=1)
plotting.niceLinPlot(ax, energies, mean_per_row, logy=False, label=r'LMA free $\Delta m_{21}^2$', color='limegreen')
ax.legend()

plt.show()