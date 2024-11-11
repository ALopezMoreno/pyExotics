import numpy as np
import arviz as az
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from scipy import stats, optimize
from graphing import plotting
from scipy.ndimage.filters import gaussian_filter
import matplotlib.ticker as ticker
from scipy.spatial import ConvexHull
import cmasher as cmr
from matplotlib.transforms import Affine2D
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import warnings
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Disable all warnings
warnings.filterwarnings("ignore")

# Create a custom legend handler
class DiagonalBoxLegendHandler(object):
    def __init__(self, color, label):
        self.color = color
        self.label = label

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        rect = plt.Rectangle([x0, y0], width, height, color='white', lw=1, ec='black', fc=self.color)
        line = Line2D([x0, x0 + width], [y0, y0 + height], color='black', lw=1)
        handlebox.add_artist(rect)
        handlebox.add_artist(line)
        return handlebox
def overlay_contours(data, ax, fill=False, **plotKwargs):
    hull = ConvexHull(data)
    ordered_points = data[hull.vertices]
    # Extract x and y coordinates from ordered points
    x, y = ordered_points[:, 0], ordered_points[:, 1]

    # Close the contour by adding the first point at the end
    sorted_x = np.append(x, x[0])
    sorted_y = np.append(y, y[0])

    t = np.arange(0, len(sorted_x))
    cs_x = CubicSpline(t, sorted_x)
    cs_y = CubicSpline(t, sorted_y)

    # Generate the smoothed perimeter points
    num_points = int(len(x) * 2.7 )  # Adjust as needed
    smoothed_t = np.linspace(0, len(sorted_x) - 1, num_points)
    smoothed_x = cs_x(smoothed_t)
    smoothed_y = cs_y(smoothed_t)

    # Plot the smoothed contour

    if not fill:
        ax.plot(smoothed_x, smoothed_y, **plotKwargs)
    else:
        polygon = np.asarray([smoothed_x, smoothed_y]).T
        pol = plt.Polygon(polygon, closed=False, fill=True, **plotKwargs)
        ax.add_patch(pol)


#////////////////////////////////////////////////////////////////#
trace = az.from_netcdf("fitResults/fitOutput_HNL_dec_wKM_long.nc")
#trace = az.from_netcdf("fitResults/fitOutput_HNL_wKM_a31.nc")

parameter_names = [r'$\sin^2\theta_{12}$', r'$\alpha_{11}$', r'$\Delta m_{21}^2$']#, r'$\sin^2\theta_{13}$'] #,r'$\alpha_{11}$' r'$\sin^2\theta_{14}$'] , r'$\sin^2\theta_{13}$'
# Turn trace into np array
posterior_samples = np.empty((len(parameter_names), trace.posterior[parameter_names[0]].shape[1]))
for i, name in enumerate(parameter_names):
    posterior_samples[i, :] = trace.posterior[name].values[0]
parameter_array = posterior_samples.T
parameter_array[:, 1] = 1 - parameter_array[:, 1]**2

print('drawing samples')
n_samples = 10**2
selected_samples = parameter_array[np.random.choice(parameter_array.shape[0], n_samples, replace=False)]
np.savetxt("fitResults/out.txt", selected_samples)


# bestFit = np.loadtxt('fitResults/fitOUtput_HNL_wKM_bestfit_a31zero.txt')
bestFit = np.loadtxt('fitResults/bestfit.txt')
bestFit_useful = [bestFit[0], bestFit[1], bestFit[2]] #, bestFit[1]
#bestFit_useful[1] = 1 - bestFit_useful[1]**2
print(bestFit_useful)
#////////////////////////////////////////////////////////////////#

def assign_values_with_kde(data, quantiles, possible_values):
    # Calculate KDE
    kde = stats.gaussian_kde(data)

    # Extract x values corresponding to quantiles
    x_values = np.concatenate(([data.min()], np.percentile(data, quantiles * 100), [data.max()]))
    interval_centers = (x_values[:-1] + x_values[1:]) / 2
    # Calculate KDE values for x values
    kde_values = kde(interval_centers)

    # Find interval with highest KDE value
    highest_kde_interval = np.argmax(kde_values)
    if highest_kde_interval == 1 and quantiles[0] < 0.01:
        highest_kde_interval = 0
    elif highest_kde_interval == len(kde_values) - 1 and quantiles[0] > 0.99:
        highest_kde_interval -= 1

    # Assign values based on KDE peak interval and adjacent intervals
    assigned_values = []
    for i in range(len(interval_centers)):
        distance_to_peak = abs(i - highest_kde_interval)
        assigned_value = possible_values[min(distance_to_peak, len(possible_values)-1)]
        assigned_values.append(assigned_value)


    return assigned_values

def find_all_roots(func, interval, num_points=10, tol=1e-6):
    roots = []
    a, b = interval
    step_size = (b - a) / num_points

    prev_sign = None
    for i in range(num_points):
        x = a + i * step_size
        f_x = func(x)

        if f_x == 0:
            roots.append(x)
            continue

        curr_sign = f_x > 0
        if prev_sign is not None and curr_sign != prev_sign:
            result = optimize.root_scalar(func, bracket=(x - step_size, x), method='brentq', xtol=tol)
            if result.converged:
                roots.append(result.root)
        prev_sign = curr_sign

    return roots
def calculate_quantiles(data, x_value):
    quantile = np.sum(data < x_value) / len(data)
    return quantile

def remove_similar_values(input_array, percentage_threshold):
    sorted_array = np.sort(input_array)
    mask = np.diff(sorted_array) <= sorted_array[:-1] * percentage_threshold
    filtered_array = sorted_array[~np.concatenate(([False], mask))]
    return filtered_array

def get_quantiles(data):
    hist, edges = np.histogram(data, bins=50)
    interpolation_factor = 5
    # Calculate new bin edges
    new_bin_edges = np.linspace(edges[0], edges[-1], len(edges) * interpolation_factor)
    # Interpolate histogram values
    bin_centres = (edges[:-1] + edges[1:]) / 2
    hist_big = np.interp(new_bin_edges, bin_centres, hist)
    new_bin_centres = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2
    # Calculate bin centers
    # Sort hist
    sorted_indices = np.argsort(hist_big)[::-1]
    sorted_hist = hist_big[sorted_indices]
    sorted_centres = new_bin_edges[sorted_indices]
    cumulative_dist = np.cumsum(sorted_hist)
    desired_percentages = [0.9, 0.99, 0.9973]

    # Find points for each desired percentage

    yValues = []
    xValues = []
    for desired_percentage in desired_percentages:
        indices = np.where(cumulative_dist >= desired_percentage * cumulative_dist[-1])
        yValues.append(sorted_hist[np.min(indices)] / np.max(hist_big))
        xValues.append(sorted_centres[np.min(indices)])

    kde = stats.gaussian_kde(data)
    x_vals = np.linspace(new_bin_edges.min(), new_bin_edges.max(), 10**3)  # Generate x values for evaluation
    kde_values = kde(x_vals)  # Evaluate the KDE at each x value
    max_kde_value = np.max(kde_values)

    def get_x_intercept(target_y):
        def difference_function(x):
            return kde(x) - target_y * max_kde_value

        interval = [np.min(data), np.max(data)]
        intersection_x_values = find_all_roots(difference_function, interval)
        return intersection_x_values

    quants = []
    for i, y in enumerate(yValues):
        quantiles = get_x_intercept(y)
        for k in quantiles:
            quants.append(k)

    flat_values = np.sort(np.asarray(quants).flatten())
    mid = (np.min(bin_centres) + np.max(bin_centres)) / 2
    unique_x_values = remove_similar_values(flat_values, 0.1*mid)

    for i, value in enumerate(unique_x_values):
        unique_x_values[i] = calculate_quantiles(data, value)
    print('here go the percentiles')
    print(unique_x_values)
    return(unique_x_values)


def get_bestFit_3D(kde):
    x, y, z = np.mgrid[
              parameter_array[:, 0].min():parameter_array[:, 0].max():100j,
              parameter_array[:, 1].min():parameter_array[:, 1].max():100j,
              parameter_array[:, 2].min():parameter_array[:, 2].max():100j
              ]
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    density = kde(positions)
    density = density.reshape(x.shape)

    max_density_idx = np.argmax(density)
    bestFit_point = np.array([x.ravel()[max_density_idx], y.ravel()[max_density_idx], z.ravel()[max_density_idx]])
    return bestFit_point

num = 10**3
color = 'dodgerblue'  # 'dodgerblue'
linecolor = None
bstcolor = 'white'
line = False




parameter_names = [r'$\sin^2\theta_{12}$', r'$\alpha_{11}$',  r'$\Delta m_{21}^2$'] #, r'$\alpha_{11}$',

# Turn trace into np array
posterior_samples = np.empty((len(parameter_names), trace.posterior[parameter_names[0]].shape[1]))
for i, name in enumerate(parameter_names):
    posterior_samples[i, :] = trace.posterior[name].values[0]
parameter_array = posterior_samples.T
parameter_array[:, 1] = 1 - parameter_array[:, 1]**2


print('drawing samples')
n_samples = 3*10**4
selected_samples = parameter_array[np.random.choice(parameter_array.shape[0], n_samples, replace=False)]
# np.savetxt("fitResults/out.txt", selected_samples)
# THIS IS FOR THE 3NU PLOTTING
#buffer = np.random.uniform(0,1, n_samples)
#selected_samples = np.vstack([selected_samples.T, [buffer]])
# calculate best fit point
print('calculating ]best fit point using smaller sample')
kde = gaussian_kde(selected_samples.T)
bestFit_point = get_bestFit_3D(kde)
print(bestFit_point)
bestFit_useful = bestFit_point

np.savetxt('fitResults/bestfit.txt', [bestFit_point[0], bestFit_point[1], bestFit_point[2]]) # bestFit_point[1]

az.plot_pair(trace, var_names=parameter_names)


fig, axes2 = plt.subplots(3, 3, figsize=(8, 8), dpi=150)#, constrained_layout=True)  # Adjust the size as needed

axes2[1, 0].hist([4,10**3], bins=1,  label=r'LMA fit (3$\nu$)', color=plotting.fade_color_to_white(color, 0.3), density=True)

#  with $\Delta m_{21}^2$ prior

az.plot_pair(trace, var_names=parameter_names,
             ax=axes2, marginals=True, scatter_kwargs={'color':'white'},
             marginal_kwargs={ 'fill_kwargs':{'color':color, 'alpha':0.5}})
                              #'quantiles':[0.5-0.9973/2,
                              #             0.5-0.9545/2,
                              #             0.5-0.6827/2,
                              #             0.5+0.6827/2,
                              #             0.5+0.9545/2,
                              #             0.5+0.9973/2], 'fill_kwargs':{'alpha':[ 1,0.5,0.3,0.1,0.3,0.5,1],
                              #                                           'color':color}})

coords = [(1,0),(2,0),(2,1)]
coordsh = [(0, 0), (1, 1), (2, 2)]

for coord in coords:
    # Reset x-axis and y-axis tick formatters
    axes2[coord].xaxis.set_ticks_position('both')
    axes2[coord].yaxis.set_ticks_position('both')

    hist, xedges, yedges = np.histogram2d(parameter_array[:,coord[1]], parameter_array[:,coord[0]], bins=80)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    xgrid, ygrid = np.meshgrid(xcenters, ycenters)

    confidence_intervals = [0, 0.683, 0.955, 0.9973]#, 0.99]#5, 0.99]  # , 0.9973 -this is 3rd sigmatotal_counts = np.sum(hist)
    non_zero_hist_values = hist[hist > 0]
    sorted_hist_values = np.sort(non_zero_hist_values.ravel())
    cumsum = np.cumsum(sorted_hist_values)
    total = cumsum[-1]
    levels = []
    for i in confidence_intervals:
        abs_diff = np.abs(cumsum/total - (1-i))
        index = np.argmin(abs_diff)
        levels.append(sorted_hist_values[index])
    levels = np.sort(levels)
    hist = gaussian_filter(hist, sigma=1.5)

    axes2[coord].contourf(xgrid, ygrid, hist.T, levels=levels,
                          cmap=plotting.create_sequential_colormap( 'white', color),
                          locator=ticker.LogLocator())

    if linecolor is not None:
        axes2[coord].contour(xgrid, ygrid, hist.T, levels=levels,
                             colors=linecolor, linewidths=1,)
    print('coordinates are ', coord)
    print(bestFit_useful[coord[1]], bestFit_useful[coord[0]])
    axes2[coord].scatter([bestFit_useful[coord[1]]],[bestFit_useful[coord[0]]], color=bstcolor, marker='+', s=150)

    if line:
        axes2[coord].axvline(x=bestFit_useful[coord[1]], color=bstcolor, linewidth=2)
        axes2[coord].axhline(y=bestFit_useful[coord[0]], color=bstcolor, linewidth=2)

    axes2[coord].set_box_aspect(1)
    plotting.makeTicks(axes2[coord], xdata=parameter_array[:, coord[1]], ydata=parameter_array[:, coord[0]], allsides=True,
    xnumber=5, ynumber=5)

    axes2[coord].xaxis.label.set_size(23)  # Adjust the font size for x-axis label
    axes2[coord].yaxis.label.set_size(23)

for coord in coordsh:
    axes2[coord].clear()
    quant = get_quantiles(parameter_array[:, coord[0]])
    if coord != (2,2):
        rot = False
    else:
        rot = True
        print(coord)
    possible_values = [0.95, 0.6, 0.2, 0.1]  # [0.1, 0.4, 0.9]  # [1, 0.5, 0.3, 0.1, 0]
    assigned_values = assign_values_with_kde(parameter_array[:, coord[0]], quant, possible_values)

    axes2[coord] = az.plot_kde(parameter_array[:, coord[0]], rotated=rot, quantiles=quant,
                              fill_kwargs={'alpha': assigned_values,
                                                     'color':color}, ax=axes2[coord], plot_kwargs={'alpha':1, 'linewidth':2},  label='This fit')


    axes2[coord].set_box_aspect(1)
    if coord != (2, 2):
        axes2[coord].xaxis.set_ticks_position('bottom')
        axes2[coord].yaxis.set_ticks_position('none')
        plotting.makeTicks(axes2[coord], xdata=parameter_array[:, coord[0]],
                           xnumber=5, ynumber=5)
        if line:
            axes2[coord].axvline(x=bestFit_useful[coord[1]], color=bstcolor, linewidth=2)

    else:
        axes2[coord].xaxis.set_ticks_position('none')
        axes2[coord].yaxis.set_ticks_position('left')
        plotting.makeTicks(axes2[coord], ydata=parameter_array[:, coord[0]],
                           xnumber=5, ynumber=5)
        if line:
            axes2[coord].axhline(y=bestFit_useful[coord[1]], color=bstcolor, linewidth=2)

    axes2[coord].set_xticklabels([])
    axes2[coord].set_yticklabels([])

#fix dm21 label
axes2[2, 0].set_ylabel(r'$\Delta m_{21}^2 \times 10^{-5}$ eV$^2$')
axes2[0, 0].set_ylabel('')
#axes2[2, 2].set_xlabel(r'$\Delta m_{21}^2 \times 10^{-5}$ eV$^2$')
axes2[1, 1].set_xticklabels([])
axes2[1, 1].set_yticklabels([])

# Fix alpha labels
axes2[1, 0].set_ylabel(r'$1-\alpha_{11}^2$')
axes2[2, 1].set_xlabel(r'$1-\alpha_{11}^2$')
# Overlay contours from other experiments
# //////////////////////////////////////////////////////////////////// #
solar1 = np.genfromtxt('contours/contour1.csv', delimiter=',')
solar1[:, 0] = np.sin(np.arctan(np.sqrt(solar1[:, 0])))**2
solar1[:, 1] = solar1[:, 1]*10**-4

solar2 = np.genfromtxt('contours/contour2.csv', delimiter=',')
solar2[:, 0] = np.sin(np.arctan(np.sqrt(solar2[:, 0])))**2
solar2[:, 1] = solar2[:, 1]*10**-4

solar3 = np.genfromtxt('contours/contour3.csv', delimiter=',')
solar3[:, 0] = np.sin(np.arctan(np.sqrt(solar3[:, 0])))**2
solar3[:, 1] = solar3[:, 1]*10**-4

bestFitSolar = np.genfromtxt('contours/bestFitSolar.csv', delimiter=',')
bestFitSolar[0] = np.sin(np.arctan(np.sqrt(bestFitSolar[0])))**2
bestFitSolar[1] = bestFitSolar[1]*10**-4


kamLAND1 = np.genfromtxt('contours/kamLAND1.csv', delimiter=',')
kamLAND1[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND1[:, 0])))**2
kamLAND1[:, 1] = kamLAND1[:, 1]*10**-4

kamLAND2 = np.genfromtxt('contours/kam2.csv', delimiter=',')
kamLAND2[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND2[:, 0])))**2
kamLAND2[:, 1] = kamLAND2[:, 1]*10**-4

kamLAND3 = np.genfromtxt('contours/kamland3.csv', delimiter=',')
kamLAND3[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND3[:, 0])))**2
kamLAND3[:, 1] = kamLAND3[:, 1]*10**-4

bestFitkamLAND = np.genfromtxt('contours/bestFitkamLAND.csv', delimiter=',')
bestFitkamLAND[0] = np.sin(np.arctan(np.sqrt(bestFitkamLAND[0])))**2
bestFitkamLAND[1] = bestFitkamLAND[1]*10**-4

global1 = np.genfromtxt('contours/global1.csv', delimiter=',')
global1[:, 0] = np.sin(np.arctan(np.sqrt(global1[:, 0])))**2
global1[:, 1] = global1[:, 1]*10**-4

global2 = np.genfromtxt('contours/global2.csv', delimiter=',')
global2[:, 0] = np.sin(np.arctan(np.sqrt(global2[:, 0])))**2
global2[:, 1] = global2[:, 1]*10**-4

global3 = np.genfromtxt('contours/global3.csv', delimiter=',')
global3[:, 0] = np.sin(np.arctan(np.sqrt(global3[:, 0])))**2
global3[:, 1] = global3[:, 1]*10**-4

bestFitglobal = np.genfromtxt('contours/bestFitglobal.csv', delimiter=',')
bestFitglobal[0] = np.sin(np.arctan(np.sqrt(bestFitglobal[0])))**2
bestFitglobal[1] = bestFitglobal[1]*10**-4

sno2nu3 = np.genfromtxt('contours/sno2nu_3sigma.csv', delimiter=',')
sno2nu3[:, 0] = np.sin(np.arctan(np.sqrt(sno2nu3[:, 0])))**2
sno2nu3[:, 1] = sno2nu3[:, 1]*10**-4


contcol = 'magenta'
kamcol = 'black'
globcol = 'orange'
snocol = 'green'

overlay_contours(solar1, axes2[2, 0], color=plotting.fade_color_to_white(contcol, 0.), lw=1.75, ls='dashed')  # linewidth=10)
overlay_contours(solar2, axes2[2, 0], color=plotting.fade_color_to_white(contcol, 0.), lw=1.75, label='Solar')
overlay_contours(solar3, axes2[2, 0], color=plotting.fade_color_to_white(contcol, 0.), lw=1.75)
axes2[2, 0].plot(bestFitSolar[0], bestFitSolar[1], color=contcol, linestyle='', marker='d', markersize=5)

overlay_contours(kamLAND1, axes2[2, 0], color=plotting.fade_color_to_white(kamcol, 0.2), lw=1.75, ls='dashed')  # linewidth=10)
overlay_contours(kamLAND2, axes2[2, 0], color=plotting.fade_color_to_white(kamcol, 0.2), lw=1.75, label='KamLAND')
overlay_contours(kamLAND3, axes2[2, 0], color=plotting.fade_color_to_white(kamcol, 0.2), lw=1.75)
axes2[2, 0].plot(bestFitkamLAND[0], bestFitkamLAND[1], color=kamcol, linestyle='', marker='s', markersize=5)

overlay_contours(global1, axes2[2, 0], fill=False, color=plotting.fade_color_to_white(globcol, 0.), lw=1.75, ls='dashed')
overlay_contours(global2, axes2[2, 0], fill=False, color=plotting.fade_color_to_white(globcol, 0.), lw=1.75, label='Solar + KamLAND')
overlay_contours(global3, axes2[2, 0], fill=False, color=plotting.fade_color_to_white(globcol, 0), lw=1.75, )  # linewidth=10)
axes2[2, 0].plot(bestFitglobal[0], bestFitglobal[1], color=globcol, linestyle='', marker='X', markersize=7)

#overlay_contours(sno2nu3, axes2[1, 0], fill=False, color=plotting.fade_color_to_white(snocol, 0.), lw=1.75, label='SNO 2nu')
print('pepito')
#axes2[2, 0].set_ylim(0*10**-5, 18.5*10**-5)
#axes2[2, 2].set_ylim(0*10**-5, 18.5*10**-5)

#axes2[2, 0].set_ylim(4*10**-5, 11*10**-5)
#axes2[2, 2].set_ylim(4*10**-5, 11*10**-5)
edges = axes2[0, 0].get_ylim()

#axes2[0, 0].set_ylim(-edges[1]/30, edges[1])

#axes2[0, 0].set_xlim(0.18, 0.4)
#axes2[2, 0].set_xlim(0.18, 0.4)

#axes2[2, 0].set_ylim(6*10**-5, 9*10**-5)
#axes2[2, 2].set_ylim(6*10**-5, 9*10**-5)
# //////////////////////////////////////////////////////////////////// #


def y_formatter(x, pos):
    return f'{x * 10**5:.1f}'  # Format the tick value by multiplying with 10^5 and displaying as a float

def y_formatter2(x, pos):
    return f'{x:.2f}'

# Apply the custom formatter to the y-axis
#axes2[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
#axes2[2, 1].xaxis.set_major_formatter(ticker.FuncFormatter(y_formatter2))
axes2[2, 0].yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))

# plot PDG constraint
xmin, xmax = axes2[0, 0].get_xlim()
ymin, ymax = axes2[0, 0].get_ylim()
x = np.linspace(xmin, xmax, 5*10**2)
y = norm.pdf(x, 0.307, 0.0129961)
y = y / np.max(y) * ymax * 0.95  # this last number is just a normalisation factor
#
axes2[0, 0].plot(x, y, linestyle='-', label=r'PDG $\sin^2\theta_{12}$ constraint', color='tab:red', linewidth=2)




plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=-0.1)  # Adjust the values as needed

fig.legend(loc='upper right', bbox_to_anchor=(1, 0.95), fontsize='15')  # Adjust the position as needed





# Extract parameter values from the thinned trace
# Convert the list of parameter values to a NumPy array
#parameter_array = np.array(parameter_values).T  # Transpose to get samples x parameters
#np.savetxt("fitResults/out.txt", parameter_array)
plt.show()
print('done!')

