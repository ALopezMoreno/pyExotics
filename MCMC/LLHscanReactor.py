import matplotlib.colors
import numpy as np

import nonUnitary.sterileOsc
from vanilla import oscillatorBase
from tqdm import tqdm
from graphing import plotting
import matplotlib.pyplot as plt
from scipy.special import factorial
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib.ticker import ScalarFormatter
from scipy.stats import poisson

#  This script creates a toy reactor experiment based in
#  DayaBay. It is useful for studying the reactor constraint.
class DayaBay_Data():
    def __init__(self, Energies, smearing=0, mode=0):
        self.E = Energies * 10**-3
        self.smearConstant = smearing
        self.mode = mode

        if self.mode:
            self.oscFar = nonUnitary.sterileOsc.oneSterile_oscillator(1.663, self.E[0], smearing=[.8, self.smearConstant], block=True)
            self.oscNear = nonUnitary.sterileOsc.oneSterile_oscillator(0.400, self.E[0], smearing=[.8, self.smearConstant], block=True)
        else:
            self.oscFar = oscillatorBase.Oscillator(1.663, self.E[0], smearing=[.8, self.smearConstant], block=True)
            self.oscNear = oscillatorBase.Oscillator(0.400, self.E[0], smearing=[.8, self.smearConstant], block=True)

        self.sin2_2th13 = 0.085
        self.th14 = 0
        self.set_th13()

        self.P = np.zeros(len(Energies))
        self.Pnear = np.zeros(len(Energies))
        self.Pfar = np.zeros(len(Energies))

    def set_th13(self):
        self.oscFar.theta13 = np.arcsin(0.5*np.sqrt(self.sin2_2th13))
        self.oscFar.setPMNS()

        self.oscNear.theta13 = np.arcsin(0.5*np.sqrt(self.sin2_2th13))
        self.oscNear.setPMNS()

    def set_th14(self, th14):
        self.th14=th14
        if self.mode:
            self.oscFar.theta14 = self.th14
            self.oscFar.set_nonUnitary()
            self.oscFar.build_mixing_matrix()

            self.oscNear.theta14 = self.th14
            self.oscNear.set_nonUnitary()
            self.oscNear.build_mixing_matrix()

        else:
            print('ERROR: this object is in unitary mode')
            exit()

    def propagate(self):

        for i in range(len(self.E)):
            self.oscFar.E = self.E[i]
            self.oscNear.E = self.E[i]
            self.Pnear[i] = self.oscNear.getOsc(0, 0)
            self.Pfar[i] = self.oscFar.getOsc(0, 0)
            self.P[i] = self.Pfar[i] / self.Pnear[i]

#  Model takes in an energy kernel and a unitary-non unitary set up
#  Then propagates for a given set of parameters
class Model():

    def __init__(self, e_kernel, data, mode=0):
        self.energies = e_kernel*10**-3
        self.mode = mode
        self.th13 = 0.13
        self.m_ee = 2.6*10**-3
        self.L = 0
        self.f_solar = 0
        self.data = data
        self.setParams()

    def setParams(self, sin2th13=0, eMass=0):
        self.th13 = np.arcsin(np.sqrt(sin2th13))
        self.esmear = np.outer(self.energies, np.random.uniform(1-0.5*.8, 1+0.5*.8, self.data.smearConstant))
        self.del_ee = 1.27 * eMass * self.L / self.esmear
        self.f_solar = np.sin(np.arcsin(np.sqrt(0.307)) * 2) ** 2 * np.sin(1.27 * 7.5*10**(-5) * self.L / self.esmear) ** 2

    def getProb(self):
        if self.mode:
            a = 1
        else:

            probT = 1 - np.cos(self.th13)**4 * self.f_solar\
                    - np.sin(2*self.th13)**2 * np.sin(self.del_ee)**2

            prob = probT.sum(axis=1) / self.data.smearConstant
        return prob

    def getLikelihood(self, sin2th13, eMass):

        self.L = 0.400
        self.setParams(sin2th13, eMass)
        probNear = self.getProb()
        self.L = 1.663
        self.setParams(sin2th13, eMass)
        probFar = self.getProb()
        probTotal = probFar/probNear
        #discretise data:
        events_per_bin = 1500
        data = np.rint(self.data.P*events_per_bin).astype(int)
        e = np.ones(len(data))*np.e

        # get the poisson probability mass function (ie prob of data given real value)
        poissonProbs = poisson.pmf(data, probTotal*events_per_bin)
        LLH = np.prod(poissonProbs) * 10**len(self.energies)
        #LLH = np.sum((probTotal-self.data)**2) / len(probTotal)
        return LLH

#%% Test to see the shape of the oscillograms
energies = np.linspace(0.5, 8, 25)
for k in np.linspace(0, np.pi/8, 4):
    th14 = k
    test = DayaBay_Data(energies, smearing=10**3, mode=1)
    test.set_th14(th14)
    test.propagate()

    fig, ax = plt.subplots(1, 1)
    plotting.niceLinPlot(ax, energies, test.Pnear, logx=False, logy=False)
    ax.set_ylim(0.5, 1.2)
    plotting.niceLinPlot(ax, energies, test.Pfar, logx=False, logy=False, color='cyan')
    plotting.niceLinPlot(ax, energies, test.P, logx=False, logy=False, color='r')
    ax.axhline(y=1, color='black')

    testModel = Model(energies, test)
    #fig2, ax2 = plt.subplots(1, 1)
    testModel.L = 1.663
    testModel.setParams(sin2th13=0.02, eMass=2.6*10**-3)

    Pfar = testModel.getProb()

    testModel.L = 0.400
    testModel.setParams(sin2th13=0.2, eMass=2.6*10**-3)
    Pnear = testModel.getProb()
    plt.title('observed probabilities. Smearing = 20\%.   '+ r'$\theta_{14}=$' + str(th14))
    #print(testModel.getProb())
    #plotting.niceLinPlot(ax, energies, Pfar/Pnear, logx=False, logy=False, color='orange')
    plt.show()
    plt.close()


#%% 2D LLH scan
nbins = 100
energies = np.linspace(0.3, 7.5, 27)
data = DayaBay_Data(energies, smearing=10**4)
data.propagate()

fittingModel = Model(energies, data)

sin2th13 = np.linspace(0., 0.04, nbins)
eeM = np.linspace(1.5*10**-3, 3.5*10**-3, nbins)
area = [sin2th13[0], sin2th13[-1], eeM[0], eeM[-1]]

# X, Y = meshgrid(sin2th13, eeM)
llh_base = np.zeros((nbins, nbins))

for i in tqdm(range(nbins)):
    for j in range(nbins):
        llh_base[i][j] = fittingModel.getLikelihood(sin2th13[i], eeM[j])

# normalisation stuff;
norms = np.linalg.norm(llh_base)
llh = llh_base / norms
# get contours:
values = np.sort(llh.flatten())
cumsum = np.cumsum(values)
cumsum = cumsum / np.amax(cumsum)

confidence_levels = []
for i in [1-0.997, 1-0.955, 1-0.683]:
    diff = np.absolute(cumsum - i)
    index = diff.argmin()
    confidence_levels.append(values[index])

fig, ax = plt.subplots(1, 1, dpi=200)
im = ax.imshow(np.flip(llh.T, 0), cmap=plotting.parula_map,
               extent=area)
               #norm=matplotlib.colors.LogNorm(vmin=10**-8, vmax=np.amax(llh)))
colorbar(im)
title('Fake reactor experiment Likelihood')
ax.set_aspect('auto', adjustable='datalim')
ax.set_xlabel(r'$\sin^2\theta_{13}$')
ax.set_ylabel(r'$\Delta m_{ee}^2$')

ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.axvline(x=0.02125, color='r')
cset = ax.contour(llh.T, confidence_levels, linewidths=2, cmap='Greys_r',
                  extent=area, norm=matplotlib.colors.LogNorm(vmin=10**-4.5, vmax=np.amax(llh)))

plt.savefig("images/reactorConstraint_toyExperiment_nominal.png", )
show()