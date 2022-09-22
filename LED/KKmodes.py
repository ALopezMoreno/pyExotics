# This is a script for calculating the masses of Kaluza-Klein modes given a Dirac Mass and compactification ratio
from scipy.optimize import brentq
import numpy as np
from vanilla import oscillatorBase as osc

dm21_2 = 7.42 * 10 ** (-5)
dm31_2 = 2.51 * 10 ** (-3)  # Mass square differences


class KKtower:

    def __init__(self, n, mD, R, inverted=False, approx=False):  # n=integer. nD,R=double
        self.nodes = n  # number of nodes to calculate
        self.compRadius = R  # Compactification radius in eV^(-1)
        self.massScale = mD  # Value of the lightest Dirac Mass
        self.IO = inverted
        self.approx = approx

        if self.IO == True:
            self.diracMass2 = self.massScale
            self.diracMass0 = np.sqrt(self.diracMass2 ** 2 + dm31_2)
            self.diracMass1 = np.sqrt(self.diracMass0 ** 2 + dm21_2)

        else:
            self.diracMass0 = self.massScale
            self.diracMass1 = np.sqrt(self.diracMass0 ** 2 + dm21_2)
            self.diracMass2 = np.sqrt(self.diracMass0 ** 2 + dm31_2)

        # Trascendental equations to solve
        self.func0 = lambda x: (
                x - np.pi * (self.diracMass0 * self.compRadius) ** 2 * np.cos(np.pi * x) / np.sin(np.pi * x))
        self.func1 = lambda x: (
                x - np.pi * (self.diracMass1 * self.compRadius) ** 2 * np.cos(np.pi * x) / np.sin(np.pi * x))
        self.func2 = lambda x: (
                x - np.pi * (self.diracMass2 * self.compRadius) ** 2 * np.cos(np.pi * x) / np.sin(np.pi * x))

        if self.approx == True:
            self.__get_approx_modes()
        else:
            self.__get_modes()  # calculate the modes
        self.__get_matrix()  # calculate the mixing matrix from the nodes

    def update(self, n, mD, R):
        self.nodes = n
        self.massScale = mD
        self.compRadius = R
        if self.IO == True:
            self.diracMass2 = self.massScale
            self.diracMass0 = np.sqrt(self.diracMass2 ** 2 + dm31_2)
            self.diracMass1 = np.sqrt(self.diracMass0 ** 2 + dm21_2)

        else:
            self.diracMass0 = self.massScale
            self.diracMass1 = np.sqrt(self.diracMass0 ** 2 + dm21_2)
            self.diracMass2 = np.sqrt(self.diracMass0 ** 2 + dm31_2)
        if self.approx == True:
            self.__get_approx_modes()
        else:
            self.__get_modes()
        self.__get_matrix()

    # %% PRIVATE
    def __get_modes(self):
        self.Emasses0 = np.zeros(self.nodes)  # solutions to the trascendental equation
        self.Emasses1 = np.zeros(self.nodes)
        self.Emasses2 = np.zeros(self.nodes)
        for i in range(self.nodes):
            self.Emasses0[i] = brentq(self.func0, i+0.000000000001, i + 0.5)
            self.Emasses1[i] = brentq(self.func1, i+0.000000000001, i + 0.5)
            self.Emasses2[i] = brentq(self.func2, i+0.000000000001, i + 0.5)
        # There is (precisely) one solution in each interval [n, n+1/2] but mantissa errors suck
        self.masses0 = self.Emasses0 / self.compRadius
        self.masses1 = self.Emasses1 / self.compRadius
        self.masses2 = self.Emasses2 / self.compRadius

    def __get_approx_modes(self):
        self.masses0 = np.zeros(self.nodes)  # solutions to the trascendental equation
        self.masses1 = np.zeros(self.nodes)
        self.masses2 = np.zeros(self.nodes)
        self.masses0[0] = self.diracMass0 * (1 - (np.pi / 6) * (self.diracMass0 * self.compRadius) ** 2)
        self.masses1[0] = self.diracMass1 * (1 - (np.pi / 6) * (self.diracMass1 * self.compRadius) ** 2)
        self.masses2[0] = self.diracMass2 * (1 - (np.pi / 6) * (self.diracMass2 * self.compRadius) ** 2)
        for i in range(1,self.nodes):
            self.masses0[i] = i / self.compRadius * (1 + (self.diracMass0*self.compRadius)**2/i**2)
            self.masses1[i] = i / self.compRadius * (1 + (self.diracMass1*self.compRadius)**2/i**2)
            self.masses2[i] = i / self.compRadius * (1 + (self.diracMass2*self.compRadius)**2/i**2)

    def __get_matrix(self):
        self.V = np.zeros((3, self.nodes))
        for i in range(self.nodes):
            self.V[0, i] = np.sqrt(
                2 / (1 + (np.pi * self.diracMass0 * self.compRadius) ** 2 + (self.masses0[i] / self.diracMass0) ** 2))
            self.V[1, i] = np.sqrt(
                2 / (1 + (np.pi * self.diracMass1 * self.compRadius) ** 2 + (self.masses1[i] / self.diracMass1) ** 2))
            self.V[2, i] = np.sqrt(
                2 / (1 + (np.pi * self.diracMass2 * self.compRadius) ** 2 + (self.masses2[i] / self.diracMass2) ** 2))


class KKoscillator(osc.Oscillator):
    def __init__(self, L, E, tower, smear=None, inverted=0):
        osc.Oscillator.__init__(self, L, E, smearing=smear, inverted=inverted)
        self.KKmodes = tower


    def getOsc(self, alpha, beta, antineutrino=False):
        if self.Esmear == 0:
            tempE = self.E
        else:
            if not self.block:
                tempE = np.random.normal(self.E, 1 / 3 * self.Esmear * self.E, self.nsmear)
            else:
                tempE = np.random.uniform(self.E - 0.5 * self.Esmear, self.E + 0.5 * self.Esmear, self.nsmear)

        terms = 0
        if antineutrino == True:
            for i in range(self.KKmodes.nodes):

                terms0 = ((self.PMNS[alpha, 0]).conjugate() * self.PMNS[beta, 0]).conjugate() * self.KKmodes.V[0, i] ** 2 * \
                              np.exp( -self.KKmodes.masses0[i] ** 2 * self.L * 1.27 * 2j / tempE)
                terms1 = ((self.PMNS[alpha, 1]).conjugate() * self.PMNS[beta, 1]).conjugate() * self.KKmodes.V[1, i] ** 2 * \
                              np.exp( -self.KKmodes.masses1[i] ** 2 * self.L * 1.27 * 2j / tempE)
                terms2 = ((self.PMNS[alpha, 2]).conjugate() * self.PMNS[beta, 2]).conjugate() * self.KKmodes.V[2, i] ** 2 * \
                              np.exp( -self.KKmodes.masses2[i] ** 2 * self.L * 1.27 * 2j / tempE)

                terms += terms0 + terms1 + terms2

        else:
            for i in range(self.KKmodes.nodes):
                terms0 = (self.PMNS[alpha, 0]).conjugate() * self.PMNS[beta, 0] * self.KKmodes.V[0, i] ** 2 * \
                         np.exp(-self.KKmodes.masses0[i] ** 2 * self.L * 1.27 * 2j / tempE)
                terms1 = (self.PMNS[alpha, 1]).conjugate() * self.PMNS[beta, 1] * self.KKmodes.V[1, i] ** 2 * \
                         np.exp(-self.KKmodes.masses1[i] ** 2 * self.L * 1.27 * 2j / tempE)
                terms2 = (self.PMNS[alpha, 2]).conjugate() * self.PMNS[beta, 2] * self.KKmodes.V[2, i] ** 2 * \
                         np.exp(-self.KKmodes.masses2[i] ** 2 * self.L * 1.27 * 2j / tempE)

                terms += terms0 + terms1 + terms2

        P = np.abs(terms) ** 2
        if self.nsmear == 0:
            P_avg = P
        else:
            P_avg = np.sum(P) / self.nsmear
        return P_avg
