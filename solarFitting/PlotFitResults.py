import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from graphing import plotting

trace = az.from_netcdf("fitResults/fitOutput_woKM.nc")


#plt.rcParams['figure.subplot.hspace'] = 0.3  # Adjust the value to your preference



parameter_names = [r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$', r'$\Delta m_{21}^2$'] #, r'$\sin^2\theta_{14}$']
num = 1000


# Thinning the posterior samples
thinned_trace = az.extract(trace, var_names=parameter_names, num_samples=num)
axes = az.plot_posterior(trace, var_names=parameter_names)

# Extract parameter values from the thinned trace
parameter_values = [thinned_trace[param_name] for param_name in parameter_names]
# Convert the list of parameter values to a NumPy array
parameter_array = np.array(parameter_values).T  # Transpose to get samples x parameters
np.savetxt("fitResults/out.txt", parameter_array)
plt.show()
"""
xmin, xmax = axes[0][0].get_xlim()
x = np.linspace(xmin, xmax, 10**2)
y = norm.pdf(x, 0.022, 0.0007)
axes[0][0].plot(x, y, linestyle='dashed', label=r'Prior', color='black')

xmin, xmax = axes[2][0].get_xlim()
x = np.linspace(xmin, xmax, 10**2)
y = norm.pdf(x, 0.307, 0.0129961)
axes[2][0].plot(x, y, linestyle='dashed', label=r'T2K solar constraint', color='red')
plotting.makeTicks(axes[0][0], ynumber=0, sci=False)
plotting.makeTicks(axes[1][0], ynumber=0, sci=False)
plotting.makeTicks(axes[2][0], ynumber=0)
axes[0][0].legend()
axes[1][0].legend()
plt.show()
"""
