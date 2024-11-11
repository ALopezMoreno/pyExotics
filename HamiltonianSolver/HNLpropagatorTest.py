# This was created to numerically solve the hamiltonian proposed in New J. Phys. 19 (2017) 093005

import sympy
import numpy as np
from math import comb

class HNLpropagator:

    def __init__(self, mixpars, IH, antinu):

        if len(mixpars) != 10:
            print('ERROR: you must provide 4 unitary mixing parameters and 6 non-unitary mixing parameters')
            exit()

        else:
            self.Upars = mixpars[0:4]
            self.nonUpars = mixpars[4:]


            self.L = 1    # Default L value (Km)
            self. E = 1   # Default E value (GeV)
            self. Ne = 0  # Default electron density
            self. Nn = 0  # Default neutron density
            self.antinu = antinu  # Is this a neutrino or an antineutrino?
            self.IH = IH # Are we in the normal or inverted hierarchy?
            self.U = self.setPMNS(3, self.Upars)  # Unitary matrix
            self.A = self.setA(self.nonUpars)  # Lower triangular non-u contribution

            self.masses = []
            self.applyNominalHierarchy()
            self.ham = np.zeros((3, 3), dtype=complex)
            self.vacham = np.zeros((3, 3), dtype=complex)
            self.matham = np.zeros((3, 3), dtype=complex)
            self.mixingMatrix = np.zeros((3, 3), dtype=complex)
            self.eigenvals = np.asarray([0, 0, 0])

    def applyNominalHierarchy(self):
        temp_masses = self.masses[3:]
        dm21 = 7.53 * 10 ** (-5)
        if self.IH:
            dm23 = - 2.494 * 10 ** (-3)
        else:
            dm23 = 2.494 * 10 ** (-3)

        self.masses = [0, dm21, dm21 + dm23]
        self.masses.extend(temp_masses)
    def setA(self, pars):
        A = np.zeros((3, 3), dtype=complex)
        A[0, 0] = pars[0]
        A[1, 0] = pars[1]
        A[1, 1] = pars[2]
        A[2, 0] = pars[3]
        A[2, 1] = pars[4]
        A[2, 2] = pars[5]

        return A

    def setPMNS(self, generations, mixingPars):
        # First, make sure the number of mixing parameters fits the dimensionality of the matrix:
        expectedAngles = comb(generations, 2)
        expectedParNumber = (generations - 1) ** 2
        if len(mixingPars) != expectedParNumber:
            print('ERROR: we expect ' + str(expectedParNumber) + ' mixing parameters for a ' + str(
                generations) + 'x' +
                  str(generations) + ' mixing matrix but ' + str(len(mixingPars)) + ' were specified')
            exit()

        # Then we create rotation matrices with phases attached to them
        rotations = []
        phasesApplied = 0

        for k in range(expectedAngles):
            mixing = np.identity(generations, dtype=complex)

            # ordering of parameters is 12, 13, 23, 14, 24, 34, 15, 25, 35, 45..., phi1, phi2, phi3...
            # in this ordering, the second index of the angle is the largest integer k for which
            # the sum 1+...+(p-1) is smaller or equal to the place of the angle in mixingPars "n"
            # luckily, we know that p-1 must be smaller than sqrt(n) (and n = k+1 because indexing starts at 0)
            # the first index of the angle is simply the place of the angle - said sum + 1
            if k == 0:
                index1 = 0
                index2 = 1
            else:
                guess = int(np.ceil(np.sqrt(k + 1)))
                mySum = np.sum(np.arange(1, guess + 1))

                while (mySum >= k + 1):
                    guess = guess - 1
                    mySum = np.sum(np.sum(np.arange(1, guess + 1)))

                index2 = int(guess + 1)
                index1 = int(k - mySum)

            mixing[index1, index1] = np.cos(mixingPars[k])
            mixing[index2, index2] = np.cos(mixingPars[k])
            mixing[index1, index2] = +np.sin(mixingPars[k])
            mixing[index2, index1] = -np.sin(mixingPars[k])

            # Now we add the complex Phases:
            if phasesApplied < (expectedParNumber - expectedAngles):
                if index2 + 1 > 2 and phasesApplied < index2:
                    # we apply the phase
                    mixing[index1, index2] *= np.exp(-mixingPars[phasesApplied + expectedAngles] * 1j)
                    mixing[index2, index1] *= np.exp(mixingPars[phasesApplied + expectedAngles] * 1j)
                    phasesApplied += 1

            rotations.append(mixing)
        if generations == 4:
            rotations[0], rotations[1], rotations[2], rotations[3], rotations[4], rotations[5] = (rotations[5],
                                                                                                  rotations[4],
                                                                                                  rotations[3],
                                                                                                  rotations[2],
                                                                                                  rotations[1],
                                                                                                  rotations[0])
        # Now we multiply the rotation matrices together, according to the PDG ordering (thus the swapping):
        if generations == 3:
            rotations[0], rotations[2] = rotations[2], rotations[0]

        myPMNS = rotations[0]

        for rotation in rotations[1:]:
            myPMNS = np.matmul(myPMNS, rotation)
        if generations == 5:
            print(np.asarray(rotations).real)
            print(myPMNS.real)
            exit()

        # This should give us a PMNS :D. Take in account mode
        if self.antinu:
            return np.conj(myPMNS)
        else:
            return myPMNS

    def setHam(self):

        N = np.matmul(self.A, self.U)
        N_H = np.transpose(np.conjugate(N))

        M = np.diag(self.masses)
        NN_dag = np.matmul(N, N_H)

        G_f = 5.4489e-5 # THIS IS IN UNITS OF N_e / N_a!!!

        V = np.zeros((3, 3))
        V[0, 0] = (2*self.Ne - self.Nn) * self.E * G_f * np.sqrt(2)
        V[1, 1] = - self.Nn * self.E * G_f * np.sqrt(2)
        V[2, 2] = - self.Nn * self.E * G_f * np.sqrt(2)

        self.vacHam = np.matmul(np.matmul(N, M), N_H)
        self.matHam = np.matmul(np.matmul(NN_dag, V), NN_dag)

        if self.antinu:
            self.ham = self.vacHam - self.matHam
        else:
            self.hamiltonian = self.vacHam + self.matHam

    def getOrderedEigenObjects(self, inputMatrix, vacuum=False):
        unordered_Eigenvals, unordered_MixingMatrix = np.linalg.eig(inputMatrix)
        unsorted_eigvals = np.real(unordered_Eigenvals)
        sorting_indices = np.argsort(unsorted_eigvals)

        sorted_eigvals = np.asarray(unsorted_eigvals, dtype=float)[sorting_indices]
        sorted_MixingMatrix = np.asarray(unordered_MixingMatrix, dtype=complex)[:, sorting_indices]

        if not vacuum:
            # Apply sign normalization
            for i in range(sorted_MixingMatrix.shape[1]):
                first_component_computed = sorted_MixingMatrix[0, i]  # First component of the computed eigenvector
                first_component_target = self.vacHam[i, 0].conjugate()  # First component of the target eigenvector
                phase_diff = np.angle(first_component_target) - np.angle(first_component_computed)  # Phase difference
                sorted_MixingMatrix[:, i] *= np.exp(-1j * phase_diff)  # Apply phase normalization

        # We now have to invert the matrix because we have the matrix expressing flav in terms of mass coords
        # And the PMNS is defined as mass in terms of flav coords
        return sorted_eigvals, (sorted_MixingMatrix)

    def update(self):
        self.setHam()
        self.eigenvals, self.mixingMatrix = self.getOrderedEigenObjects(self.hamiltonian)