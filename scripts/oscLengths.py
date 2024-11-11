import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphing import plotting
from matplotlib import ticker
import matplotlib


matplotlib.use('pgf')
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'pdf.fonttype': 42,  # Embed fonts in the PDF
    'ps.fonttype': 42,
    'pgf.rcfonts': False,  # Do not use rc settings in pgf preamble
})


def get_osclength(Delm2):
    LoverE = np.pi / (2 * Delm2 * 1.26693281)
    return LoverE

mass = np.logspace(-3.99, 2.99, num=500)
stripes = np.sin(np.log10(mass))

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(6, 5))

oscLength = get_osclength(mass**2)

# plot
plotting.niceLinPlot(ax, mass, oscLength, color='gray', linestyle='dashed', linewidth=1.5, alpha=0.5)
plotting.makeTicks(ax, mass, oscLength, allsides=True)

T2KLoverE = 295/0.6
T2Kdecoherence = 0.09/0.6

ax.axhline(T2KLoverE, color='black', linewidth=2, linestyle='dashdot', label='T2K optimal')
#ax.axhline(0.280/0.6, color='red', linestyle='dashed', linewidth=2)
#ax.axhline(T2Kdecoherence, color='black', linestyle='dotted', linewidth=2)

ax.set_ylim(10**(-3.5), 10**5.5)

ax.plot(np.sqrt(2.4*10**(-3)), get_osclength(2.4*10**(-3)), marker='^', linestyle='', color='red', markersize=7, label='Atm splitting')
ax.plot(np.sqrt(7*10**(-5)), get_osclength(7*10**(-5)),  marker='^', linestyle='', color='blue', markersize=7, label='Sol splitting')


# Gallium
ax.plot(np.array([np.sqrt(0.6),100]), get_osclength(np.array([0.6, 100**2])), color='orange',linewidth=2, label='Gallium allowed')

# LSND
ax.plot(np.sqrt(1.2), get_osclength(1.2), marker='^', linestyle='', color='magenta', markersize=7, label='LSND best-fit')

# Reactor shape
ax.plot(np.sqrt(1.3), get_osclength(1.3), marker='^', linestyle='', color='darkgreen', markersize=7, label='RSS best-fit')

# Shades
ax.fill_between(mass, T2KLoverE*3/0.6, 10**6, color='gray', hatch='/', alpha=0.2)
ax.fill_between(mass, 10**(-4), T2Kdecoherence, color='royalblue', hatch='/', alpha=0.2)
ax.text(10*0.6, 10**4.3, 'No oscillations at T2K', fontsize=18, ha='center', va='center')
ax.text(10**-2*5, 10**-2.3, 'Decoherent mixing at T2K', fontsize=18, ha='center', va='center')

# Arrow
arrow_start = (10*2, get_osclength(10**2))  # Starting point of the arrow (x, y)
arrow_end = (100, get_osclength(50**2))   # Ending point of the arrow (x, y)
ax.annotate('', xy=arrow_end, xytext=arrow_start, arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

dx = np.log10(arrow_end[0]) - np.log10(arrow_start[0])
dy = np.log10(arrow_end[1]) - np.log10(arrow_start[1])
angle = np.arctan2(dy, dx) * 180 / np.pi

text_position = (arrow_end[0]/1.7, arrow_end[1]*7)  # Position for the text (x, y)
rotated_text = ax.text(text_position[0], text_position[1], 'HNLs', rotation=-52.5, ha='center', va='center', fontsize=12, color='red')



ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=14)

ax.set_xlabel(r'mass (eV)', fontsize=18)
ax.set_ylabel(r'Oscillation length (Km/GeV)', fontsize=18)

ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
ax.tick_params(axis='x', which='major', pad=5)  # Increase pad to separate labels from axis

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.5)
ax.tick_params(which='both', width=1)


plt.savefig('../images/oscLengths.pdf', format='pdf', bbox_inches='tight')
plt.show()
