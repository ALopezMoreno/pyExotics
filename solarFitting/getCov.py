import numpy as np
import sys
sys.path.append('../')
import arviz as az
import matplotlib.pyplot as plt
from graphing import plotting
from matplotlib.colors import LogNorm  # Import LogNorm for logarithmic normalization
from scipy.stats import norm
from scipy.linalg import sqrtm
from scipy.stats import gaussian_kde, chi2
from scipy.interpolate import griddata

trace = az.from_netcdf("fitResults/fitOutput_HNL_wKM_a31zero.nc")

parameter_names = [r'$\sin^2\theta_{12}$', r'$\alpha_{11}$']

# Turn trace into np array
posterior_samples = np.empty((len(parameter_names), trace.posterior[parameter_names[0]].shape[1]))
for i, name in enumerate(parameter_names):
    posterior_samples[i, :] = trace.posterior[name].values[0]

parameter_array = posterior_samples
parameter_array[1, :] = 1 - parameter_array[1, :]**2

# Perform the transformation in one line and save it to 'points_transformed'
points_transformed = np.array([2 * 0.306 - parameter_array[0, :], -parameter_array[1, :]])

total = np.concatenate((parameter_array, points_transformed), axis=1)

# Get covariance
cov_matrix = np.cov(total)

print("Covariance Matrix:")
print(cov_matrix)


# Define confidence levels (1, 2, and 3 sigma)
confidence_levels = [0.683, 0.955, 0.997]
line = ['-', '--', ':']




fig,ax = plt.subplots(1, 2, dpi=200, sharey=True)

hist, xedges, yedges, image = ax[0].hist2d(parameter_array[0, :], parameter_array[1, :], bins=(80, 80),
                                        cmap='bone_r')
plotting.makeTicks(ax[0], parameter_array[0, :], parameter_array[1, :], allsides=True, xnumber=5, ynumber=5)


# Find the bin with the highest count
max_count = np.max(hist)
max_count_index_flat = np.argmax(hist)
max_count_index = np.unravel_index(max_count_index_flat, hist.shape)

# Calculate the bin center of the highest bin
x_center = (xedges[max_count_index[0]] + xedges[max_count_index[0] + 1]) / 2
y_center = (yedges[max_count_index[1]] + yedges[max_count_index[1] + 1]) / 2
mean = [x_center, y_center]

print(f"The bin center of the highest bin is ({x_center}, {y_center}), with a count of {max_count}.")

# Plot the best fit point
ax[0].plot(x_center, y_center, color='r', marker='P', markersize=7)
ax[0].set_xlabel(r'$\sin^2\theta_{12}$', fontsize=20)
ax[0].set_ylabel(r'$\sin^2\theta_{14}$', fontsize=20)



# Plot contours for each confidence level
for i, sigma in enumerate(confidence_levels):
    chi2_value = chi2.ppf(sigma, df=2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    std_devs = np.sqrt(chi2_value * eigenvalues)

    # Calculate the ellipse's parameters
    width = 2 * std_devs[0]
    height = 2 * std_devs[1]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Create and add the ellipse to the plot
    ellipse = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle, fill=False, edgecolor='red', lw=2.0,
                                             linestyle=line[i])
    ax[0].add_patch(ellipse)

# Add a colorbar to the existing plot
#colorbar = plt.colorbar(image, ax=ax)  # hist[3] corresponds to the image created by hist2d
#colorbar.ax.tick_params(labelsize=20)  # Adjust the font size as needed

ax[0].set_box_aspect(1)

#%%

# Plot 1D hist

hist, edges, img = ax[1].hist(parameter_array[1, :], bins=80, orientation='horizontal', density=True)

# Plot sigma lines
cumsum = np.cumsum(hist)
b_centres = (edges[:-1] + edges[1:]) / 2
percentile_cumsum = cumsum / cumsum[-1]
for threshold in confidence_levels:
    index = np.where(percentile_cumsum > threshold)[0][0]
    print(index)
    ax[1].axhline(y=b_centres[index], lw=2, color='red')

# plot 1d gaussian from covariance

x = np.linspace(y_center - 3 * np.sqrt(cov_matrix[1, 1]), y_center + 3 * np.sqrt(cov_matrix[1, 1]), 300)
y = (1 / (np.sqrt(2 * np.pi * cov_matrix[1, 1]))) * np.exp(-(x - y_center)**2 / (2 * cov_matrix[1, 1]))
ax[1].plot(y*2, x, color='red')
ax[1].set_box_aspect(1)
plt.tight_layout()
plt.show()