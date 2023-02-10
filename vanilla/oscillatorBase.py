# Base vanilla oscillator class, which can then be modified in each exotics scenario
import numpy as np


class Oscillator:

    def __init__(self, L, E, smearing=None, inverted=0, block=False):  # L in Km and E in GeV

        self.block = block

        self.theta12 = np.arcsin(np.sqrt(0.307))
        self.theta23 = np.arcsin(np.sqrt(0.561))
        self.theta13 = np.arcsin(np.sqrt(0.022))
        self.dcp = -1.601

        self.dm21_2 = 7.42 * 10 ** (-5)
        if inverted == 0:
            self.dm32_2 = 2.494 * 10 ** (-3)
        else:
            self.dm32_2 = - 2.494 * 10 ** (-3)

        self.dm31_2 = self.dm32_2 + self.dm21_2
        self.L = L
        self.E = E
        if smearing:
            self.Esmear = smearing[0]
            self.nsmear = smearing[1]
        else:
            self.Esmear = None
            self.nsmear = 0
        self.PMNS = np.zeros((3, 3), dtype=complex)
        self.setPMNS()

    def update(self, L, E):
        self.L = L
        self.E = E

    # Build nominal PMNS matrix from given mixing parameters
    def setPMNS(self):
        # set abbreviations
        S12 = np.sin(self.theta12)
        S23 = np.sin(self.theta23)
        S13 = np.sin(self.theta13)
        C12 = np.sqrt(1 - S12 * S12)
        C23 = np.sqrt(1 - S23 * S23)
        C13 = np.sqrt(1 - S13 * S13)
        cd = np.cos(self.dcp)
        sd = np.sin(self.dcp)
        # e row
        self.PMNS[0, 0] = complex(C12 * C13, 0)
        self.PMNS[0, 1] = complex(S12 * C13, 0)
        self.PMNS[0, 2] = complex(S13 * cd, -S13 * sd)
        # mu row
        self.PMNS[1, 0] = complex(-S12 * C23 - C12 * S23 * S13 * cd, -C12 * S23 * S13 * sd)
        self.PMNS[1, 1] = complex(C12 * C23 - S12 * S23 * S13 * cd, -S12 * S23 * S13 * sd)
        self.PMNS[1, 2] = complex(S23 * C13, 0)
        # tau row
        self.PMNS[2, 0] = complex(S12 * S23 - C12 * C23 * S13 * cd, -C12 * C23 * S13 * sd)
        self.PMNS[2, 1] = complex(-C12 * S23 - S12 * C23 * S13 * cd, -S12 * C23 * S13 * sd)
        self.PMNS[2, 2] = complex(C23 * C13, 0)

    # Calculate oscillation probability. If Esmear is non-zero, it will return the probability
    # for a randomly gaussian sampled energy with standard deviation 1/3*Esmear*E centered at E
    def getOsc(self, alpha, beta, antineutrino=False):
        if self.Esmear == 0:
            tempE = self.E
        else:
            if not self.block:
                tempE = np.random.normal(self.E, 1 / 3 * self.Esmear * self.E, self.nsmear)
            else:
                tempE = np.random.uniform(self.E - 0.5 * self.Esmear*self.E, self.E + 0.5 * self.Esmear*self.E, self.nsmear)
        if alpha == beta:
            KronD = 1
        else:
            KronD = 0

        Rterm21 = ((self.PMNS[alpha, 1]).conjugate() * self.PMNS[beta, 1] * self.PMNS[alpha, 0] *
                   (self.PMNS[beta, 0]).conjugate()).real * np.sin(1.26693281 * self.dm21_2 * self.L / tempE) ** 2

        Rterm32 = ((self.PMNS[alpha, 2]).conjugate() * self.PMNS[beta, 2] * self.PMNS[alpha, 1] *
                   (self.PMNS[beta, 1]).conjugate()).real * np.sin(1.26693281 * self.dm32_2 * self.L / tempE) ** 2

        Rterm31 = ((self.PMNS[alpha, 2]).conjugate() * self.PMNS[beta, 2] * self.PMNS[alpha, 0] *
                   (self.PMNS[beta, 0]).conjugate()).real * np.sin(1.26693281 * self.dm31_2 * self.L / tempE) ** 2

        Iterm21 = ((self.PMNS[alpha, 1]).conjugate() * self.PMNS[beta, 1] * self.PMNS[alpha, 0] *
                   (self.PMNS[beta, 0]).conjugate()).imag * np.sin(1.26693281 * 2 * self.dm21_2 * self.L / tempE)

        Iterm32 = ((self.PMNS[alpha, 2]).conjugate() * self.PMNS[beta, 2] * self.PMNS[alpha, 1] *
                   (self.PMNS[beta, 1]).conjugate()).imag * np.sin(1.26693281 * 2 * self.dm32_2 * self.L / tempE)

        Iterm31 = ((self.PMNS[alpha, 2]).conjugate() * self.PMNS[beta, 2] * self.PMNS[alpha, 0] *
                   (self.PMNS[beta, 0]).conjugate()).imag * np.sin(1.26693281 * 2 * self.dm31_2 * self.L / tempE)

        if antineutrino == True:
            P = KronD - 4 * (Rterm21 + Rterm32 + Rterm31) - 2 * (Iterm21 + Iterm32 + Iterm31)
        else:
            P = KronD - 4 * (Rterm21 + Rterm32 + Rterm31) + 2 * (Iterm21 + Iterm32 + Iterm31)

        P_avg = np.sum(P) / np.max([self.nsmear, 1])

        return P_avg
