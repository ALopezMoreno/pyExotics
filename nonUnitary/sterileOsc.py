import numpy as np
from vanilla import oscillatorBase as osc

#Implements one heavy neutral lepton mixing
class oneSterile_oscillator(osc.Oscillator):
    def __init__(self, L, E, smear=None, inverted=0, block=False):
        osc.Oscillator.__init__(self, L, E, smearing=smear, inverted=inverted)

        # Always instance with a trivial non-unitary part:
        # New mixing angles:
        self.mixM = None
        self.theta14 = 0
        self.theta24 = 0
        self.theta34 = 0
        # New complex phases:
        self.phi14 = 0
        self.phi24 = 0
        self.phi34 = 0

        self.A = np.identity(3, dtype=complex)
        self.set_nonUnitary()
        self.build_mixing_matrix()

    # Build non-unitary lower-triangular "A" matrix. Calculate A^2
    def set_nonUnitary(self):

        S14 = np.sin(self.theta14)
        S24 = np.sin(self.theta24)
        S34 = np.sin(self.theta34)
        C14 = np.sqrt(1 - S14 * S14)
        C24 = np.sqrt(1 - S24 * S24)
        C34 = np.sqrt(1 - S34 * S34)

        # The diagonal entries are well-behaved
        self.A[0, 0] = complex(C14, 0)
        self.A[1, 1] = complex(C24, 0)
        self.A[2, 2] = complex(C34, 0)
        # The off-diagonals have complex components:
        self.A[1, 0] = complex(-np.cos(self.phi24 - self.phi14), np.sin(self.phi24 - self.phi14)) * S14 * S24
        self.A[2, 0] = complex( - np.cos(self.phi34 - self.phi14) * S14 * C24 * S34,
                               np.sin(self.phi34 - self.phi14) * S14 * C24 * S34)
        self.A[2, 1] = complex(-np.cos(self.phi34 - self.phi24), np.sin(self.phi34 - self.phi24)) * S24 * S34

        # Since this is a lower-triangular matrix the remaining entries are zero
        # It only remains to square the matrix
        self.A2 = np.matmul(self.A, self.A)

    def build_mixing_matrix(self):
        self.mixM = np.matmul(self.A, self.PMNS)

    # Calculate oscillation probabilities. Nearly identical to the oscillatorBase class.
    def getOsc(self, alpha, beta, antineutrino=False):

        if self.Esmear == 0:
            tempE = self.E
        else:
            if not self.block:
                tempE = np.random.normal(self.E, 1 / 3 * self.Esmear * self.E, self.nsmear)
            else:
                tempE = np.random.uniform(self.E - 0.5 * self.Esmear, self.E + 0.5 * self.Esmear, self.nsmear)

        Rterm21 = ((self.mixM[alpha, 1]).conjugate() * self.mixM[beta, 1] * self.mixM[alpha, 0] *
                   (self.mixM[beta, 0]).conjugate()).real * np.sin(1.27 * self.dm21_2 * self.L / tempE) ** 2

        Rterm32 = ((self.mixM[alpha, 2]).conjugate() * self.mixM[beta, 2] * self.mixM[alpha, 1] *
                   (self.mixM[beta, 1]).conjugate()).real * np.sin(1.27 * self.dm32_2 * self.L / tempE) ** 2

        Rterm31 = ((self.mixM[alpha, 2]).conjugate() * self.mixM[beta, 2] * self.mixM[alpha, 0] *
                   (self.mixM[beta, 0]).conjugate()).real * np.sin(1.27 * self.dm31_2 * self.L / tempE) ** 2

        Iterm21 = ((self.mixM[alpha, 1]).conjugate() * self.mixM[beta, 1] * self.mixM[alpha, 0] *
                   (self.mixM[beta, 0]).conjugate()).imag * np.sin(1.27 * 2 * self.dm21_2 * self.L / tempE)

        Iterm32 = ((self.mixM[alpha, 2]).conjugate() * self.mixM[beta, 2] * self.mixM[alpha, 1] *
                   (self.mixM[beta, 1]).conjugate()).imag * np.sin(1.27 * 2 * self.dm32_2 * self.L / tempE)

        Iterm31 = ((self.mixM[alpha, 2]).conjugate() * self.mixM[beta, 2] * self.mixM[alpha, 0] *
                   (self.mixM[beta, 0]).conjugate()).imag * np.sin(1.27 * 2 * self.dm31_2 * self.L / tempE)

        if antineutrino:
            P = self.A2[alpha, beta] - 4 * (Rterm21 + Rterm32 + Rterm31) - 2 * (Iterm21 + Iterm32 + Iterm31)
        else:
            P = self.A2[alpha, beta] - 4 * (Rterm21 + Rterm32 + Rterm31) + 2 * (Iterm21 + Iterm32 + Iterm31)

        P_avg = np.sum(P) / np.max([self.nsmear, 1])
        return P_avg.real
