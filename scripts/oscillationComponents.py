import numpy as np
from graphing import plotting
from HamiltonianSolver import customPropagator
import matplotlib.pyplot as plt
import matplotlib

# Initialise propagator to generate global best-fit parameters
E = 1
L = 1
prop = customPropagator.HamiltonianPropagator(customPropagator.matterHamiltonian, L, E, False, False, density=0)

# Set mixing matrix and masses
U = prop.PMNS
masses_NH = prop.masses
prop.IH = True
prop.update()
masses_IH = prop.masses

m21_NH = masses_NH[1] - masses_NH[0]
m32_NH = masses_NH[2] - masses_NH[1]
m31_NH = masses_NH[2] - masses_NH[0]

m21_IH = masses_IH[1] - masses_IH[0]
m32_IH = masses_IH[2] - masses_IH[1]
m31_IH = masses_IH[2] - masses_IH[0]

# For testing
print('Normal hierarchy: ', m21_NH, m32_NH, m31_NH)
print('Inverse hierarchy: ', m21_IH, m32_IH, m31_IH)

LoverE = np.logspace(1, 5, 10**4)

print(U[0, 1])
m21_amp = U[0, 1].conj() * U[1, 1] * U[0, 0] * U[1, 0].conj()
m32_amp = U[0, 2].conj() * U[1, 2] * U[0, 1] * U[1, 1].conj()
m31_amp = U[0, 2].conj() * U[1, 2] * U[0, 0] * U[1, 0].conj()

term_21 = -4*np.real(m21_amp)*np.sin(m21_NH*1.27*LoverE)**2 - 2*np.imag(m21_amp)*np.sin(m21_NH*1.27*LoverE*2)
term_32 = -4*np.real(m32_amp)*np.sin(m32_NH*1.27*LoverE)**2 - 2*np.imag(m32_amp)*np.sin(m32_NH*1.27*LoverE*2)
term_31 = -4*np.real(m31_amp)*np.sin(m31_NH*1.27*LoverE)**2 - 2*np.imag(m31_amp)*np.sin(m31_NH*1.27*LoverE*2)

total = 0 + (term_21 + term_32 + term_31)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6), dpi=200)

# Plot components
plotting.niceLinPlot(ax[1], LoverE, term_32, logx=True, logy=False, color='royalblue', alpha=0.8, linewidth=2, label=r'$f(\Delta m^2_{32})$')
plotting.niceLinPlot(ax[1], LoverE, term_31, logx=True, logy=False, color='red', alpha=0.7, linewidth=2, label=r'$f(\Delta m^2_{31})$')
plotting.niceLinPlot(ax[1], LoverE, term_21, logx=True, logy=False, color='limegreen', alpha=0.8, linewidth=2, label=r'$f(\Delta m^2_{21})$')

# Plot total
plotting.niceLinPlot(ax[0], LoverE, total, logx=True, logy=False, color='blue', alpha=0.7, linewidth=2)

#Fix ticks
plotting.makeTicks(ax[1], xdata=LoverE, ydata=np.linspace(-0.1, 0.5, 10))
plotting.makeTicks(ax[0], xdata=LoverE, ydata=[0,0.6])
plt.subplots_adjust(hspace=0.03)  # You can adjust the value of hspace as needed

ax[1].tick_params(axis='x', which='major', direction='inout', length=8)
ax[1].set_xlabel(r'$L/E$ (Km/GeV)', fontsize=18)
ax[0].set_ylabel(r'$P(\nu_\mu\to\nu_e)$', fontsize=18)

ax[1].set_ylabel(r'Contribution', fontsize=18)
ax[1].legend(loc='best', fontsize=16)

#Add experiments
ax[0].axvline(x=0.35*10**3, color='gray', linewidth=160, alpha=0.2, label=r'Atmospheric')
ax[1].axvline(x=0.35*10**3, color='gray', linewidth=160, alpha=0.2)

ax[0].axvline(x=297/0.6, color='brown', linewidth=15, alpha=0.5, label=r'Accelerator')
ax[1].axvline(x=297/0.6, color='brown', linewidth=15, alpha=0.5)

ax[0].axvline(x=1.7*10**4, color='gold', linewidth=110, alpha=0.2, label=r'Reactor')
ax[1].axvline(x=1.7*10**4, color='gold', linewidth=110, alpha=0.2)

legend_proxy_circle1 = matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=15, label=r'Atm.', alpha=0.3)
legend_proxy_circle2 = matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', markersize=15, label=r'Acc.', alpha=0.65)
legend_proxy_circle3 = matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=15, label=r'Reac.', alpha=0.5)

ax[0].legend(handles=[legend_proxy_circle1,
                      legend_proxy_circle2,
                      legend_proxy_circle3], fontsize=16)

for ax2 in ax:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(1.3)

#ax[0].axvline(x=800/2, color='cyan', linewidth=3)
#ax[1].axvline(x=800/2, color='cyan', linewidth=3)

plt.savefig('../images/oscillationFactors.pdf', format='pdf', bbox_inches='tight')