# A script for fitting LMA solutions to P_ee data
import sys
sys.path.append("../")
import numpy as np
import pymc as pm
from solarFluxes import solarProbs as spr
import matplotlib.pyplot as plt
import LMAmodels
import arviz as az
from scipy.stats import norm, uniform
from graphing import plotting

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# Load data:
pp = spr.pp
Be7 = spr.Be7
pep = spr.pep
B8 = spr.B8
B8_SNO = spr.B8_SNO

energies = np.concatenate([pp["Energy [MeV]"], Be7["Energy [MeV]"], pep["Energy [MeV]"], B8["Energy [MeV]"], B8_SNO["Energy [MeV]"]])
errors = np.concatenate([pp["Err Up"], Be7["Err Up"], pep["Err Up"], B8["Err Up"], B8_SNO["Err Up"]])
data = np.concatenate([pp["Pee"], Be7["Pee"], pep["Pee"], B8["Pee"], B8_SNO["Pee"]])



# Set up the model
with pm.Model() as LMA_model:
    # Priors for unknown model parameters
    sin2th12_prior = pm.Uniform(r'$\sin^2\theta_{12}$', lower=0, upper=1)
    sin2th13_prior = pm.Normal(r'$\sin^2\theta_{13}$', mu=0.022, sigma=0.0007)
    # dm21_prior = pm.Normal(r'$\Delta m_{21}^2$', mu=7.51*10**-5, sigma=0.3*10**-5) #Use KamLAND
    dm21_prior = pm.Uniform(r'$\Delta m_{21}^2$', lower=10**-6, upper=1.5*10**-4)

    # NON UNITARY:
    #sin2th14_prior = pm.Uniform(r'$\sin^2\theta_{14}$', lower=0, upper=0.1)

    #th12 = pm.Deterministic('th12', np.arcsin(np.sqrt(sin2th12_prior)))
    #th13 = pm.Deterministic('th13', np.arcsin(np.sqrt(sin2th13_prior)))

    # Expected value of outcome
    mu = LMAmodels.LMA_solution(energies, dm21_prior, sin2th12_prior, sin2th13_prior)
    #mu = pm.Normal(r'P${ee}$', mu=LMAmodels.LMA_solution_4nu(energies, dm21_prior, sin2th12_prior, sin2th13_prior, sin2th14_prior), sigma=errors, shape=len(energies))

    # Likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=errors, observed=data)

    # Sampling
    #trace = pm.sample()
    trace = pm.sample(draws=30*10**3, chains=4, tune=5*10**3, return_inferencedata=True)  # Adjust parameters as needed

    print('now saving')
    trace.to_netcdf("fitResults/fitOutput_woKM.nc")

#map_estimate = pm.find_MAP(model=LMA_model)
#print(map_estimate)
# Plot results
plt.rcParams['figure.subplot.hspace'] = 0.6  # Adjust the value to your preference
axes = az.plot_trace(trace, figsize=(10, 18), compact=False, var_names=[r'$\sin^2\theta_{13}$', r'$\Delta m_{21}^2$',
                                                                        r'$\sin^2\theta_{14}$', r'$\sin^2\theta_{12}$'])

print('plotting')
# Overlay priors
xmin, xmax = axes[0][0].get_xlim()
x = np.linspace(xmin, xmax, 10**2)
y = norm.pdf(x, 0.022, 0.0007)
axes[0][0].plot(x, y, linestyle='dashed', label=r'Prior', color='black')
plotting.makeTicks(axes[0][0], ynumber=0, sci=False)
plotting.makeTicks(axes[1][0], ynumber=0, sci=False)
#plotting.makeTicks(axes[2][0], ynumber=0)

xmin, xmax = axes[-1][0].get_xlim()
x = np.linspace(xmin, xmax, 10**2)
y = norm.pdf(x, 0.307, 0.0129961)
axes[-1][0].axvline(x=0.307, color='b') #plot(x, y, linestyle='dashed', label=r'Prior', color='black')


plt.savefig('traces.png')
plt.show()
#az.plot_pair(trace, kind="kde")
#plt.savefig('posteriors.png')
#plt.show()
