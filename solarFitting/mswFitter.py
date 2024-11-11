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
from scipy.optimize import curve_fit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# fitting function for SNO: we estimate the coefficients by doing finite difference around the fit point
def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c
def fit_second_degree_polynomial(dm21, sin2th12, sin2th13):
    E = np.linspace(5,10,20)
    dE = 10**-4
    n_e = 93.8
    forward = LMAmodels.LMA_solution(E+dE, dm21, sin2th12, sin2th13, n_e)
    centre = LMAmodels.LMA_solution(E, dm21, sin2th12, sin2th13, n_e)
    backward = LMAmodels.LMA_solution(E-dE, dm21, sin2th12, sin2th13, n_e)
    a = 0
    for i in centre:
        a += i/len(E)

    b1 = (forward - backward)/(2*dE)
    c1 = 0.5*(forward + backward - 2*centre)/(dE**2)

    c = 0
    for i in c1:
        c += i / len(E)
    b = 0
    for i in b1:
        b += i / len(E)

    return a, b, c


# Load data:
pp = spr.pp
Be7 = spr.Be7
pep = spr.pep
B8 = spr.B8
B8_SNO = spr.B8_SNO

e_sno = [4.057, 4.143, 4.228, 4.313, 4.399, 4.484, 4.569, 4.654, 4.74, 4.825, 4.91, 4.996, 5.081, 5.166, 5.251, 5.337, 5.422, 5.507, 5.593, 5.678, 5.763, 5.849, 5.934, 6.019, 6.104, 6.19, 6.275, 6.36, 6.446, 6.531, 6.616, 6.701, 6.787, 6.872, 6.957, 7.043, 7.128, 7.213, 7.299, 7.384, 7.469, 7.554, 7.64, 7.725, 7.81, 7.896, 7.981, 8.066, 8.152, 8.237, 8.322, 8.408, 8.493, 8.578, 8.663, 8.749, 8.834, 8.919, 9.005, 9.09, 9.175, 9.261, 9.346, 9.431, 9.517, 9.602, 9.687, 9.772, 9.858, 9.943, 10.028, 10.114, 10.199, 10.284, 10.37, 10.455, 10.54, 10.626, 10.711, 10.796, 10.881, 10.967, 11.052, 11.137, 11.223, 11.308, 11.393, 11.479, 11.564, 11.649, 11.735, 11.82, 11.905, 11.991, 12.076, 12.161, 12.246, 12.332, 12.417, 12.502, 12.588, 12.673, 12.758, 12.844, 12.929, 13.014, 13.1, 13.185, 13.27, 13.356, 13.441, 13.526, 13.612, 13.697, 13.782, 13.868, 13.953, 14.038, 14.123, 14.209, 14.294, 14.379, 14.465, 14.55, 14.635, 14.721, 14.806, 14.891, 14.965]
errUp_sno_raw = [0.129, 0.125, 0.122, 0.119, 0.11499999999999999, 0.11199999999999999, 0.10899999999999999, 0.10599999999999998, 0.10299999999999998, 0.09999999999999998, 0.09699999999999998, 0.09499999999999997, 0.09199999999999997, 0.08899999999999997, 0.08700000000000002, 0.08400000000000002, 0.08100000000000002, 0.07900000000000001, 0.07600000000000001, 0.07500000000000001, 0.07200000000000001, 0.07, 0.067, 0.065, 0.062, 0.06, 0.059, 0.056999999999999995, 0.05499999999999999, 0.05299999999999999, 0.04999999999999999, 0.04899999999999999, 0.046999999999999986, 0.04500000000000004, 0.04400000000000004, 0.04300000000000004, 0.041000000000000036, 0.039000000000000035, 0.039000000000000035, 0.03700000000000003, 0.03500000000000003, 0.03500000000000003, 0.03300000000000003, 0.03200000000000003, 0.031000000000000028, 0.030000000000000027, 0.028000000000000025, 0.028000000000000025, 0.027000000000000024, 0.026000000000000023, 0.025000000000000022, 0.025000000000000022, 0.02400000000000002, 0.02400000000000002, 0.02300000000000002, 0.02300000000000002, 0.02200000000000002, 0.02100000000000002, 0.02100000000000002, 0.02100000000000002, 0.020000000000000018, 0.020000000000000018, 0.019000000000000017, 0.019000000000000017, 0.020000000000000018, 0.019000000000000017, 0.019000000000000017, 0.018000000000000016, 0.018000000000000016, 0.019000000000000017, 0.018000000000000016, 0.018000000000000016, 0.018000000000000016, 0.017000000000000015, 0.017000000000000015, 0.017000000000000015, 0.017000000000000015, 0.017000000000000015, 0.017000000000000015, 0.017000000000000015, 0.017000000000000015, 0.016000000000000014, 0.016000000000000014, 0.016000000000000014, 0.016000000000000014, 0.016000000000000014, 0.016000000000000014, 0.016000000000000014, 0.016000000000000014, 0.017000000000000015, 0.017000000000000015, 0.018000000000000016, 0.017000000000000015, 0.017000000000000015, 0.018000000000000016, 0.019000000000000017, 0.019000000000000017, 0.020000000000000018, 0.020999999999999963, 0.021999999999999964, 0.022999999999999965, 0.023999999999999966, 0.024999999999999967, 0.025999999999999968, 0.02699999999999997, 0.02799999999999997, 0.02999999999999997, 0.030999999999999972, 0.03199999999999997, 0.033999999999999975, 0.035999999999999976, 0.03799999999999998, 0.03999999999999998, 0.04099999999999998, 0.04299999999999998, 0.044999999999999984, 0.046999999999999986, 0.04899999999999999, 0.05099999999999999, 0.05299999999999999, 0.05499999999999999, 0.057999999999999996, 0.06, 0.062, 0.064, 0.066, 0.069, 0.07200000000000001, 0.07400000000000001]
data_sno = [0.267, 0.269, 0.27, 0.271, 0.272, 0.273, 0.274, 0.275, 0.276, 0.277, 0.278, 0.279, 0.28, 0.281, 0.282, 0.283, 0.284, 0.285, 0.286, 0.286, 0.287, 0.288, 0.289, 0.29, 0.291, 0.292, 0.292, 0.293, 0.294, 0.295, 0.296, 0.296, 0.297, 0.298, 0.299, 0.299, 0.3, 0.301, 0.301, 0.302, 0.303, 0.303, 0.304, 0.305, 0.305, 0.306, 0.307, 0.307, 0.308, 0.308, 0.309, 0.309, 0.31, 0.31, 0.311, 0.311, 0.312, 0.313, 0.313, 0.313, 0.314, 0.314, 0.315, 0.315, 0.315, 0.316, 0.316, 0.317, 0.317, 0.317, 0.318, 0.318, 0.318, 0.319, 0.319, 0.319, 0.319, 0.32, 0.32, 0.32, 0.32, 0.321, 0.321, 0.321, 0.321, 0.322, 0.322, 0.322, 0.322, 0.322, 0.322, 0.322, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.323, 0.322, 0.322, 0.322, 0.322, 0.322, 0.322, 0.321, 0.321, 0.321, 0.321, 0.32, 0.32, 0.32, 0.32, 0.32, 0.319, 0.319, 0.319]

errUp_sno = np.asarray(errUp_sno_raw)

ne_sno = np.ones(129) * 93.8
nn_sno = np.ones(129) * 37.9

energies = np.concatenate([pp["Energy [MeV]"], Be7["Energy [MeV]"], pep["Energy [MeV]"], B8["Energy [MeV]"], B8_SNO["Energy [MeV]"]]) #, B8_SNO["Energy [MeV]"]])
errors = np.concatenate([pp["Err Up"], Be7["Err Up"], pep["Err Up"], B8["Err Up"], np.asarray(B8_SNO["Err Up"])*3/2]) #, np.asarray(B8_SNO["Err Up"]*2)])
data = np.concatenate([pp["Pee"], Be7["Pee"], pep["Pee"], B8["Pee"], B8_SNO["Pee"]]) #, B8_SNO["Pee"]])
nes = np.concatenate([pp["Ne"], Be7["Ne"], pep["Ne"], B8["Ne"],  B8_SNO["Ne"]]) #, B8_SNO["Ne"]])
nns = np.concatenate([pp["Nn"], Be7["Nn"], pep["Nn"], B8["Nn"],  B8_SNO["Nn"]]) #, B8_SNO["Nn"]])
#print(errors)
# Set up the model
with pm.Model() as LMA_model:
    # Priors for unknown model parameters
    sin2th12_prior = pm.Uniform(r'$\sin^2\theta_{12}$', lower=0, upper=1)
    sin2th13_prior = pm.Normal(r'$\sin^2\theta_{13}$', mu=0.022, sigma=0.0007)
    dm21_prior = pm.Normal(r'$\Delta m_{21}^2$', mu=7.49*10**-5, sigma=0.2*10**-5) #Use KamLAND
    #dm21_prior = pm.Uniform(r'$\Delta m_{21}^2$', lower=10**-6, upper=3*10**-4)

    # NON UNITARY:
    #alpha11_prior = pm.Uniform(r'$\alpha_{11}$', lower=0.7, upper=1)
    #alpha31_prior = pm.Uniform(r'$\alpha_{31}$', lower=0, upper=0.00001)
    #th12 = pm.Deterministic('th12', np.arcsin(np.sqrt(sin2th12_prior)))
    #th13 = pm.Deterministic('th13', np.arcsin(np.sqrt(sin2th13_prior)))

    # Expected value of outcome
    #mu = LMAmodels.LMA_solution_4nu(energies, dm21_prior, sin2th12_prior, sin2th13_prior, alpha11_prior, N_e=nes, N_n=nns) #, sin2th14_prior)
    mu1 = LMAmodels.LMA_solution(energies, dm21_prior, sin2th12_prior, sin2th13_prior, nes)
    #mu = pm.Normal(r'P${ee}$', mu=LMAmodels.LMA_solution_4nu(energies, dm21_prior, sin2th12_prior, sin2th13_prior, sin2th14_prior), sigma=errors, shape=len(energies))
    a, b, c = fit_second_degree_polynomial(dm21_prior, sin2th12_prior, sin2th13_prior)
    
    # Likelihood for borexino points
    term0 = pm.Normal('likelihood1', mu=mu1, sigma=errors, observed=data)

    # Likelihood for SNO fit
    #term1 = pm.Normal('likelihood2', mu=[a, b, c],
    #                  sigma=[0.016+0.009, 0.007+0.0045, 0.003+0.0016],
    #                  observed=[0.3174, 0.0039, -0.001])
    #term1 = pm.Normal('term1', mu=a, sigma=0.016 + 0.009, observed=0.3174)
    #term2 = pm.Normal('term2', mu=b, sigma=0.007+0.0045, observed=0.0039)
    #term3 = pm.Normal('term3', mu=c, sigma=0.003+0.0016, observed=-0.001)

    #L2 =
    #L = pm.math.prod((likelihood1, likelihood2, likelihood3, likelihood4))
    #likelihood = (term0, term1, term2, term3)
    #pm.Potential("combined", L)

    # Sampling
    #trace = pm.sample()
    trace = pm.sample(draws=2*10**6, chains=4, tune=10**5, return_inferencedata=True)  # Adjust parameters as needed
    # Standard seems 5 * 10^5 with 10^5 for tuning


    print('now saving')
    trace.to_netcdf("fitResults/fitOutput_3nu_solarRadii_wKM_long_def.nc")

# Plot results
axes = az.plot_trace(trace, figsize=(10, 18), compact=False, var_names=[r'$\sin^2\theta_{12}$', r'$\Delta m_{21}^2$'])
plt.savefig('traces_solarRadii_wKM.png')
az.plot_pair(trace, kind="kde", var_names=[r'$\sin^2\theta_{12}$', r'$\Delta m_{21}^2$'])
plt.savefig('pairplot_solarRadii_wKM.png')



"""
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
"""

#  plt.savefig('traces.png')
#  plt.show()
#  az.plot_pair(trace, kind="kde")
#  plt.savefig('posteriors.png')
#  plt.show()
