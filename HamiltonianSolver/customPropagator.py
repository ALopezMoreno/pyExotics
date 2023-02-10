# This is a class which takes in a neutrino hamiltonian and solves the Schrödinger equation to get a propagator
# which can be fed to the experiment classes. We use various numerical solving tools for this

import sympy
import numpy as np
from math import comb

class HamiltonianPropagator:

    def __init__(self, newHamiltonian, L, E):
        # we load up the standard part of the Hamiltonian:
        # INITIALISE WITH DEFAULT PARAMETER VALUES:
        self.E = E
        self.L = L
        self.generations = 3
        self.masses = [0, np.sqrt(7.42 * 10 ** (-5)), np.sqrt(2.51 * 10 ** (-3))]
        self.mixingPars = [np.arcsin(np.sqrt(0.307)), np.arcsin(np.sqrt(0.022)), np.arcsin(np.sqrt(0.561)), -1.601]
        self.setPMNS(self.generations, self.mixingPars)

        # compute the vanilla hamiltonian
        self.setVanillaHamiltonian()

        # diagonalise it to recover the input PMNS matrix and masses (this is for sanity check purposes)
        self.vEigenvals, self.vMixingMatrix = self.getOrderedEigenObjects(self.vHam)

        # do hamiltonian stuff
        self.newHam = newHamiltonian
        self.setFullHamiltonian()
        self.eigenvals, self.mixingMatrix = self.getOrderedEigenObjects(self.hamiltonian)

        # FINISH THIS AT THE END

    # set an n by n mixing matrix from a set of input mixing parameters
    def setPMNS(self, generations, mixingPars):
        # First, make sure the number of mixing parameters fits the dimensionality of the matrix:
        expectedAngles = comb(generations, 2)
        expectedParNumber = (generations - 1) ** 2
        if len(mixingPars) != expectedParNumber:
            print('ERROR: we expect ' + str(expectedParNumber) + ' mixing parameters for a ' + str(generations) + 'x' +
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
            mixing[index1, index2] = np.sin(mixingPars[k])
            mixing[index2, index1] =-np.sin(mixingPars[k])

            # Now we add the complex Phases:
            if phasesApplied < (expectedParNumber - expectedAngles):
                if index2 + 1 > 2 and phasesApplied < index2:
                    # we apply the phase
                    mixing[index1, index2] *= np.exp(-mixingPars[phasesApplied + expectedAngles] * 1j)
                    mixing[index2, index1] *= np.exp(mixingPars[phasesApplied + expectedAngles] * 1j)
                    phasesApplied += 1

            rotations.append(mixing)

        # Now we multiply the rotation matrices together, according to the PDG ordering (thus the swapping):
        if generations >= 3:
            rotations[0], rotations[2] = rotations[2], rotations[0]

        myPMNS = rotations[0]

        for i in range(1, len(rotations)):
            myPMNS = np.matmul(myPMNS, rotations[i])

        # This should give us a PMNS :D
        self.PMNS = myPMNS

    # set the vacuum hamiltonian
    def setVanillaHamiltonian(self):

        # Some basic check
        if len(self.masses) != self.generations:
            print("ERROR: you must have the same amount of masses as neutrinos!")
            exit()

        massSquares = np.multiply(self.masses, self.masses)

        massMatrix = np.diag(massSquares)
        #print(massSquares)
        Ustar = self.PMNS.conjugate()

        self.vHam = 1 / (2 * self.E) * np.matmul(np.matmul(self.PMNS, massMatrix), Ustar.transpose())
        # (Energy must be in the same units as the masses)

    # diagonalising the hamiltonian will return the eigenvalues in arbitrary order, but we want them in order of
    # increasing masses. Hence, we need to fiddle a bit
    def getOrderedEigenObjects(self, inputMatrix):
        ham = sympy.Matrix(inputMatrix)
        unordered_MixingMatrix, unordered_Eigenvals = ham.diagonalize()
        unsorted_eigvals = np.abs(np.diag(unordered_Eigenvals))

        sorting_indices = np.argsort(unsorted_eigvals)
        sorted_eigvals = np.asarray(unsorted_eigvals, dtype=float)[sorting_indices]
        sorted_MixingMatrix = np.asarray(unordered_MixingMatrix, dtype=complex)[:, sorting_indices]

        return sorted_eigvals, sorted_MixingMatrix

    def setFullHamiltonian(self):
        self.hamiltonian = self.vHam + self.newHam

    # propagate according to the plane wave solution of the hamiltonian
    def getOsc(self, alpha, beta, antineutrino=False):
        V1 = self.mixingMatrix
        P = complex(0, 0)

        if antineutrino:
            for i in range(self.generations):
                phase = self.eigenvals[i] * self.L * 1.27 * 2
                P += V1[alpha, i].conjugate() * V1[beta, i] * np.exp(-phase * 2j)
        else:
            for i in range(self.generations):
                phase = self.eigenvals[i] * self.L * 1.27 * 2
                P += V1[alpha, i] * V1[beta, i].conjugate() * np.exp(-phase * 2j)

        pOsc = np.abs(P) ** 2
        return pOsc

    # Function to update hamiltonian if any input parameters are changed
    def update(self):

        self.setPMNS(self.generations, self.mixingPars)
        self.setVanillaHamiltonian()
        self.vEigenvals, self.vMixingMatrix = self.getOrderedEigenObjects(self.vHam)
        self.setFullHamiltonian()
        self.eigenvals, self.mixingMatrix = self.getOrderedEigenObjects(self.hamiltonian)

    def new_hamiltonian(self, new):
        self.newHam = new
        self.update()
