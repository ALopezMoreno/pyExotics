import sys
sys.path.append('../')
import numpy as np
from graphing import plotting
from vanilla import oscillatorBase
from matplotlib import pyplot as plt
from MCMC import mcmc
from graphing import plotting

def gaussian(x, mu, sig):
    chi = (x - mu) / sig
    gauss = np.exp(-0.5 * chi * chi)
    return gauss

"""
throws = 10**6

S13 = np.random.normal(0.02, 0.0007, throws)
S12 = np.random.normal(0.307, np.sqrt(0.001689), throws)
S23 = np.random.uniform(0, 1, throws)
dcp = np.random.uniform(-np.pi, np.pi, throws)

osc = oscillatorBase.Oscillator(295, 0.6)

P = np.zeros(throws)
for i in range(throws):
    osc.theta13 = np.arcsin(np.sqrt(S13[i]))
    osc.theta12 = np.arcsin(np.sqrt(S12[i]))
    osc.theta23 = np.arcsin(np.sqrt(S23[i]))
    osc.dcp = dcp[i]

    osc.setPMNS()
    P[i] = osc.getOsc(0, 0)


plt.hist(P, bins=100, color='red', density=True, alpha=0.6)
for i in range(throws):
    osc.theta13 = np.arcsin(np.sqrt(S13[i]))
    #osc.theta12 = np.arcsin(np.sqrt(S12[i]))
    #osc.theta23 = np.arcsin(np.sqrt(S23[i]))
    #osc.dcp = dcp[i]

    osc.setPMNS()
    P[i] = osc.getOsc(0, 0)

n, bins, patches = plt.hist(P, bins=100, color='blue', density=True, alpha=0.6)
plt.suptitle('reactor prior P_ee probabilities with th23, th12, dcp fixed vs T2K prior')
b_centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2

probs = np.multiply(n, bins[1:] - bins[:-1])  # height times bin width
mean = np.sum(np.multiply(b_centers, probs))
print(mean)
variance=np.sum(np.multiply(probs, (b_centers-mean)**2))
print(variance)
plt.axvline(mean, color='black', linewidth=2)
x = np.linspace(bins[0], bins[-1], 1000)
y = gaussian(x, mean, np.sqrt(variance)) * np.max(n)
plt.plot(x, y, linewidth=3,  color='green')
plt.show()
"""
mean = 0.9201315988269182
variance = 7.187129083938207e-06
True_values = [0.307, 0.561, 0.022, -1.601]

print('running mcmc')
#S12, S23, S13, dcp, a, b, c, d, e, f, logl, acceptance, steptime, nstep = np.loadtxt('../posteriors/testChainSterile.txt', skiprows=10**4, dtype=float, delimiter=",").T
#newstart=[S12[-1], S23[-1], S13[-1], dcp[-1]]
mychain = mcmc.MCMC_toy_reactor_prior([mean, np.sqrt(variance)], 1, sterile=True, startPoint=True_values)
mychain.runChain(10**5, "../posteriors/ChainSterile_"+str(sys.argv[1])+".txt", start=False)

"""
print("analysing posterior")
parula = plotting.parula_map

fig2, ax = plt.subplots(nrows=2, ncols=5, dpi=400, figsize=(22, 8))
S12, S23, S13, dcp, S14, S24, S34, p14, p24, p34, logl, acceptance, steptime, nstep = np.loadtxt('../posteriors/testChainSterile.txt', skiprows=1, dtype=float, delimiter=",").T


n, bins, patches = ax[0, 0].hist(S12, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[0, 0].set_xlabel('sin2th12')
#ax[0, 0].axvline(True_values[0], color='r')

n, bins, patches = ax[0, 1].hist(S23, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[0, 1].set_xlabel('sin2th23')
#ax[0, 1].axvline(True_values[1], color='r')

n, bins, patches = ax[1, 0].hist(S13, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[1, 0].set_xlabel('sin2th13')
x = np.linspace(0.015, 0.024, 100)
ax[1, 0].plot(x, gaussian(x, 0.02, 0.0007)*np.max(n), color='r', linewidth=2)
#ax[1, 0].axvline(True_values[2], color='r')

n, bins, patches = ax[1, 1].hist(dcp, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[1, 1].set_xlabel('dcp')
#ax[1, 1].axvline(True_values[3], color='r')

n, bins, patches = ax[0, 2].hist(S14, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[0, 2].set_xlabel('sin2th14')

n, bins, patches = ax[1, 2].hist(p14, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[1, 2].set_xlabel('phi14')

n, bins, patches = ax[0, 3].hist(S24, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[0, 3].set_xlabel('sin2th24')

n, bins, patches = ax[1, 3].hist(p24, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[1, 3].set_xlabel('phi24')

n, bins, patches = ax[0, 4].hist(S34, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[0, 4].set_xlabel('sin2th34')

n, bins, patches = ax[1, 4].hist(p34, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[1, 4].set_xlabel('phi34')

for i in [0, 1]:
    for j in [0, 4]:
        ax[i, j].set_box_aspect(1)
        ax[i, j].set_yticks([])
plt.show()

plt.plot(nstep, S14)
plt.show()
plt.plot(nstep, acceptance)
plt.show()
"""