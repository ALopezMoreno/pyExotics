from MCMC import posteriorAnalysisTools
import time

start_time = time.time()

P1 = 0.9351837317909224
P2 = 0.2247488876024471
P3 = 0.05041885305237606
P4 = 0.059292849910843465

asimov = [0.307, 0.561, 0.022, -1.601]

L = 295.2
E = 1
#oldChain = posteriorAnalysisTools.posterior('posteriors/testChain2.txt')
#endpoint_old = [oldChain.S12[-1], oldChain.S23[-1], oldChain.S13[-1], oldChain.dcp[-1]]

print('running mcmc')
#mychain = mcmc.MCMC(P1, P2, P3, P4, experiments.eProfile("T2K"), L, 0.3, startPoint=asimov)
#mychain.runChain(3*10**4, "posteriors/testChain2.txt", start=True)

print("analysing posterior")
myChain = posteriorAnalysisTools.posterior('posteriors/testChain2.txt')
#plt.hist(myChain.steptime, bins=100)
#plt.show()
myChain.set_asimov(0.307, 0.561, 0.022, -1.601)

myChain.plot_1D_pdfs()
myChain.plot_variations()
myChain.plot_acceptance()
myChain.plot_logL()

"""
parula = plotting.parula_map
True_values = [0.307, 0.561, 0.022, -1.601]

 # token value


propagator = oscillatorBase.Oscillator(L, E, smearing=[0, 0])  # Basic propagator for feeding it to experiment
exp = experiments.experiment(experiments.eProfile("T2K"), L)
P1 = exp.propagate(propagator, 0, 0)
P2 = exp.propagate(propagator, 1, 1)
P3 = exp.propagate(propagator, 1, 0)
P4 = exp.propagate(propagator, 1, 0, antineutrino=True)


print(P1, P2, P3, P4)
print('ready to run mcmc')

# Load previous chain
S12, S23, S13, dcp, logl, steptime, nstep = np.loadtxt('posteriors/testChain.txt', skiprows=1, dtype=float, delimiter=",").T

# Initiate from last chain and run
mychain = mcmc.MCMC(P1, P2, P3, P4, experiments.eProfile("T2K"), L, 0.045, startPoint=[S12[-1], S23[-1], S13[-1], dcp[-1]])
mychain.runChain(5*10**4, "posteriors/testChain.txt", start=True)

# Plot posteriors
fig2, ax = plt.subplots(nrows=2, ncols=2, dpi=400, figsize=(8, 8))
S12, S23, S13, dcp, logl, steptime, nstep = np.loadtxt('posteriors/testChain.txt', skiprows=1, dtype=float, delimiter=",").T


n, bins, patches = ax[0, 0].hist(S12, bins=80, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[0, 0].set_xlabel('sin2th12')
ax[0, 0].axvline(True_values[0], color='r')

n, bins, patches = ax[0, 1].hist(S23, bins=80, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[0, 1].set_xlabel('sin2th23')
ax[0, 1].axvline(True_values[1], color='r')

n, bins, patches = ax[1, 0].hist(S13, bins=80, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[1, 0].set_xlabel('sin2th13')
ax[1, 0].axvline(True_values[2], color='r')

n, bins, patches = ax[1, 1].hist(dcp, bins=80, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, density=False)
n = n.astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(parula(n[i]/max(n)))
ax[1, 1].set_xlabel('dcp')
ax[1, 1].axvline(True_values[3], color='r')

for i in [0, 1]:
    for j in [0, 1]:
        ax[i, j].set_box_aspect(1)

plt.show()

nPoints = 1000
L = np.linspace(0,10000, nPoints)   #np.logspace(0, 4, nPoints)


P = np.ones(nPoints)
for i in range(1, nPoints):
    exp.L = L[i]
    P[i] = exp.propagate(propagator, 1, 1)

# PLOTTING
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400, figsize=(5.5, 5))
colors = [plotting.parula_map_r(i) for i in np.linspace(0, 1, 14)]

ax.axhline(1, color='black')
#plotting.niceLinPlot(ax, L, P1, logy=False, marker='.', markersize=1, ls='', color='g')
plotting.niceLinPlot(ax, L,  P, logy=False, logx=False, marker='.', markersize=1, ls='', color='b', alpha=1)

ax.axvline(295.2, color='r') #295.2
#ax.axvline(810 / 1.8, color='g')
ax.set_ylabel("P")
ax.set_xlabel("L")
ax.set_box_aspect(1)
plt.show()

vals = experiments.eProfile("T2K")
fig2, ax2 = plt.subplots(nrows=1, ncols=1, dpi=400, figsize=(5.5, 5))
ax2.bar(x=vals.numuBE[:-1], height=vals.numu, width=np.diff(vals.numuBE), align='edge', fc='blue', alpha=0.6)
#ax2.set_yscale('log')
ax2.set_xlim([0,2.5])
ax2.set_ylabel("Flux (arbitrary units)")
ax2.set_xlabel("E (GeV)")
plt.show()
"""

print('done. Time in min:')
print((time.time() - start_time) / 60.)