import numpy as np
import sys
sys.path.append("../")
from graphing import plotting
from HamiltonianSolver import customPropagator
from tqdm import tqdm
import matplotlib.pyplot as plt
from solarFluxes import solarProbs as spr
import LMAmodels
import scipy.stats as sp
from scipy.interpolate import interp1d

# LOAD BOREXINO DATA
pp = spr.pp
Be7 = spr.Be7
pep = spr.pep
B8 = spr.B8
B8_SNO = spr.B8_SNO
B8_SNO2 = spr.B8_SNO


energyMin = np.log10(0.12)
energyMax = np.log10(20)
nEnergies = 1000
nGens = 4

th12 = np.arcsin(np.sqrt(0.307)) #0.308
th13 = np.arcsin(np.sqrt(0.022)) # this is the default
S14 = 0.0
S24 = 0.0
S34 = 0.0
Hij = 0

th14 = np.arcsin(np.sqrt(S14))
th24 = np.arcsin(np.sqrt(S24))
th34 = np.arcsin(np.sqrt(S34))

# Make list of energies to loop through
energies = np.logspace(energyMin, energyMax, nEnergies)

# Calculate LMA solutions
dm12 = (7.51 * 10 ** (-5))
probsLMA2 = LMAmodels.LMA_solution(energies, 10*10**-5, 0.27, 0.022, 93.8)
probsLMA = LMAmodels.LMA_solution(energies, dm12, 0.307, 0.022, 93.8)

# Plot LMA solutions
fig, ax = plt.subplots(dpi=150, figsize=(8, 3.5))

# Get fit results
#sin2th12, a11, dm21, sin2th13 = np.loadtxt('fitResults/out.txt').T # Non-u
sin2th12, a11, dm21 = np.loadtxt('fitResults/out.txt').T
b_12, b_a11, b_m21 = np.loadtxt('fitResults/bestfit.txt')
# a11,
sin2th13 = np.ones(len(sin2th12))*0.022

datas = np.empty((len(energies), len(dm21)))
for i in range(len(sin2th12)):
    # datas[:, i] = LMAmodels.LMA_solution_4nu(energies, dm21[i],sin2th12[i], sin2th13[i], a11[i] ) #, a11[i], 0) # non U version
    datas[:, i] = LMAmodels.LMA_solution(energies, dm21[i], sin2th12[i], sin2th13[i], 93.8)
    #plotting.niceLinPlot(ax, energies, datas[:, i], logy=False, color='r', alpha=0.05, linewidth=1)

# Calculate mean and standard deviation for each row
# b_a11 = np.sqrt(1-b_a11)

mean_per_row = np.empty(nEnergies)
for i, energy in enumerate(energies):
    mean_per_row[i] = LMAmodels.LMA_solution(energy, b_m21, b_12, 0.022, 93.8)
    # mean_per_row[i] =  LMAmodels.LMA_solution_4nu(energy, b_m21, b_12, 0.022, np.sqrt(1-b_a11))

#std_per_row = np.std(datas, axis=1)
intervals = np.empty((nEnergies, 2))

#Plot a few cross sections
for i, slice in enumerate(datas):
    hist, binEdges = np.histogram(slice, bins=30)
    hist = sp.rv_histogram((hist, binEdges))
    intervals[i, :] = hist.interval(0.6827)
    #print(intervals[i, :])

plt.fill_between(energies, intervals[:, 0], intervals[:, 1], color='limegreen', alpha=0.2)

ax.set_ylim(0,0.8)
plotting.makeTicks(ax, xdata=energies)
ax.set_ylabel(r'P$_{ee}$', fontsize=18)
ax.set_xlabel(r'E (MeV)', fontsize=18)


# Plot survival probs
colors = ['orange', 'green', 'red', 'purple', 'blue']
plt.errorbar(pp["Energy [MeV]"], pp["Pee"], yerr=[pp["Err Down"], pp["Err Up"]], fmt='.', capsize=0, ecolor=colors[0], color=colors[0])
plt.errorbar(Be7["Energy [MeV]"], Be7["Pee"], yerr=[Be7["Err Down"], Be7["Err Up"]], fmt='.', capsize=0, ecolor=colors[1], color=colors[1])
plt.errorbar(pep["Energy [MeV]"], pep["Pee"], yerr=[pep["Err Down"], pep["Err Up"]], fmt='.', capsize=0, ecolor=colors[2], color=colors[2])
plt.errorbar(B8["Energy [MeV]"], B8["Pee"], yerr=[B8["Err Down"], B8["Err Up"]], fmt='.', capsize=0, ecolor=colors[3], color=colors[3])
#plt.errorbar(B8_SNO["Energy [MeV]"], B8_SNO["Pee"], yerr=[B8_SNO["Err Down"], B8_SNO["Err Up"]], fmt='.', capsize=0, ecolor=colors[4], color=colors[4])


# SNO data we treat slightly different
x_values = np.asarray(B8_SNO2["Energy [MeV]"])
y_values = np.asarray(B8_SNO2["Pee"])
y_err_down = np.asarray(B8_SNO2["Err Down"])
y_err_up = np.asarray(B8_SNO2["Err Up"])
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
        label_x = var["Energy [MeV]"][1]
        label_y = var["Pee"][0] - var["Err Up"][0] - 0.1  # Adjust the value to control label position
    
    plt.annotate(labels[i],
                 xy=(label_x, label_y),
                 xytext=(5, 10), textcoords="offset points",
                 ha="center", fontsize=18, color=colors[i], weight='bold')

plotting.niceLinPlot(ax, energies, probsLMA, logy=False, color='b', label=r'NuFit best fit', linestyle='dashed', linewidth=1.5)
plotting.niceLinPlot(ax, energies, probsLMA2, logy=False, color='orange', linestyle='dashed', linewidth=1.5)
plotting.niceLinPlot(ax, energies, mean_per_row, logy=False, label=r'HNL fit', color='limegreen')
ax.legend(fontsize=18, loc='lower left')
plt.tight_layout()
plt.savefig('survivalProbs_test.png')
plt.show()
