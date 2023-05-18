# This is a class which takes in a neutrino hamiltonian and solves the Schrödinger equation to get a propagator
# which can be fed to the experiment classes. We use various numerical solving tools for this
# Also contains utility methods for dealing with hamiltonians

import sympy
import numpy as np
from math import comb
from functools import reduce
from multiprocessing import Pool, cpu_count
import copy
import inspect
from scipy.optimize import root
from scipy.optimize import linear_sum_assignment
from typing import Callable, List, Tuple, Set

# ------  THIS IS FOR DEBUGGING - COMMENT OR ADD AS REQUIRED ----- #
def print_caller():
    # Get the frame of the caller of print_caller (plus however many levels you need)
    frame = inspect.currentframe().f_back.f_back
    # Get the filename and line number of the caller of print_caller
    filename, line_number, function_name, lines, index = inspect.getframeinfo(frame)
    # Print the line of code that called the function that called print_caller
    print(f"Caller: {lines[index].strip()}")
# ---------------------------------------------------------- #


class HamiltonianPropagator:

    def __init__(self, newHamiltonian, L, E, IH, antinu, *args, **kwargs):
        # newHamiltonian must be an energy dependent function. In particular,
        # it must be of the form newHamiltonian(E, *args)
        #IH, antinu = boolean
        # we load up the standard part of the Hamiltonian:
        # INITIALISE WITH DEFAULT PARAMETER VALUES:
        self.E = E
        self.L = L
        self.IH = IH
        self.antinu = antinu
        self.generations = 3
        self.masses = []
        self.applyNominalHierarchy()
        self.mixingPars = [np.arcsin(np.sqrt(0.307)), np.arcsin(np.sqrt(0.022)), np.arcsin(np.sqrt(0.561)), -1.601]
        self.setPMNS(self.generations, self.mixingPars)

        # compute the vanilla hamiltonian
        self.setVanillaHamiltonian()

        # diagonalise it to recover the input PMNS matrix and masses (this is for sanity check purposes)
        self.eigenOrder = np.array(range(self.generations))
        self.vEigenvals, self.vMixingMatrix = self.getOrderedEigenObjects(self.vHam)

        # do hamiltonian stuff
        self.hamBounds = []
        self.hamArgs = args
        self.hamKwargs = kwargs
        self.functHam = newHamiltonian
        self.newHam = self.functHam(self.E, *self.hamArgs, **self.hamKwargs)
        # We do net set the full hamiltonian until the full form of the (non-default) hamiltonian has been set
        # self.setFullHamiltonian()

        #maybe chillax on this for the time being:
        # self.eigenvals, self.mixingMatrix = self.getOrderedEigenObjects(self.hamiltonian)

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

        # This should give us a PMNS :D. Take in account mode
        if self.antinu:
            self.PMNS = myPMNS.conjugate()
        else:
            self.PMNS = myPMNS

    # set the vacuum hamiltonian
    def setVanillaHamiltonian(self):

        # Some basic check
        if len(self.masses) != self.generations:
            print("ERROR: you must have the same amount of masses as neutrinos!")
            print("Number of masses given: " + str(len(self.masses)))
            print("Number of neutrino states: " + str(self.generations))
            exit()

        massSquares = np.multiply(self.masses, self.masses)

        massMatrix = np.diag(massSquares)
        Ustar = self.PMNS.conjugate()

        self.vHam = np.matmul(np.matmul(self.PMNS, massMatrix), Ustar.transpose())
        # (Energy must be in the same units as the masses)

    # diagonalising the hamiltonian will return the eigenvalues in arbitrary order, but we want them in order of
    # increasing masses. Hence, we need to fiddle a bit
    def getOrderedEigenObjects(self, inputMatrix, vacuum=False):
        #print(inputMatrix)
        self.eigenOrder = np.array(range(self.generations))

        # ham = sympy.Matrix(inputMatrix)
        unordered_Eigenvals, unordered_MixingMatrix = np.linalg.eig(inputMatrix)
        unsorted_eigvals = np.real(unordered_Eigenvals)
        sorting_indices = np.argsort(unsorted_eigvals)

        sorted_eigvals = np.asarray(unsorted_eigvals, dtype=float)[sorting_indices]
        sorted_MixingMatrix = np.asarray(unordered_MixingMatrix, dtype=complex)[:, sorting_indices]

        if not vacuum:
            # Apply sign normalization
            for i in range(sorted_MixingMatrix.shape[1]):
                first_component_computed = sorted_MixingMatrix[0, i]  # First component of the computed eigenvector
                first_component_target = self.vHam[0, i]  # First component of the target eigenvector
                phase_diff = np.angle(first_component_target) - np.angle(first_component_computed)  # Phase difference
                sorted_MixingMatrix[:, i] *= np.exp(-1j * phase_diff)  # Apply phase normalization

            #indexes = np.where(np.abs(np.imag(self.vHam)) < 10**-18)[1]
            #mask = np.where(np.abs(np.imag(self.vHam)) > 10**-18)
            #for col_index in indexes:
            #    temp = sorted_MixingMatrix
            #    temp[mask] = self.vHam[mask]
            #    column = temp[:, col_index]
            #    complex_element = self.vHam[:, col_index]
            #    sign_check = np.sign(np.real(complex_element)) == np.sign(np.real(column))
            #    if not np.all(sign_check):
            #        sorted_MixingMatrix[:, col_index] *= -1

        #return sorted_eigvals[self.eigenOrder], sorted_MixingMatrix[:, self.eigenOrder]
        return sorted_eigvals, sorted_MixingMatrix


    def __get_dimensionality(self, func, hamInputs, **kwargs):
        # Get list of positional arguments for input function
        inputs = [self.E]
        inputs.extend(hamInputs)
        # Get list of positional arguments for input function
        func_args = [p.name for p in inspect.signature(self.functHam).parameters.values() if
                     p.default is inspect.Parameter.empty]

        inputs_pos = [inputs[i] for i in range(len(inputs)) if func_args[i] not in kwargs]

        # Evaluate function using input values and record output
        output = func(*inputs_pos, **kwargs)

        # Vary each input value and record output for each variation
        output_variations = []
        for i in range(len(inputs_pos)):
            inputs_variation = inputs_pos.copy()
            inputs_variation[i] += 1e-6  # small perturbation to vary input value
            output_variation = func(*inputs_variation, **kwargs)
            output_variations.append(output_variation)

        output_variations = np.array(output_variations)

        # Determine dimensionality of output space using SVD
        U, s, V = np.linalg.svd(output_variations)
        rank = np.sum(s > 1e-15)
        output_dim = rank

        # Determine independence of input arguments
        input_variations = output_variations.T
        U, s, V = np.linalg.svd(input_variations)
        rank = np.sum(s > 1e-10)
        input_dim = rank
        independent_args = [func_args[i] for i in range(len(inputs)) if func_args[i] not in kwargs][:input_dim]
        dependent_args = [arg for arg in func_args if arg not in independent_args]

        return input_dim, independent_args, dependent_args


    def setFullHamiltonian(self):
        if self.antinu:
            self.hamiltonian = self.vHam - self.newHam
        else:
            self.hamiltonian = self.vHam + self.newHam

    # propagate according to the plane wave solution of the hamiltonian
    def getAmps(self, alpha, beta):
        V1 = self.mixingMatrix
        P = complex(0, 0)

        for i in range(self.generations):
            phase = self.eigenvals[i] * self.L * 1.27 * 2
            P += V1[alpha, i].conjugate() * V1[beta, i] * np.exp(-phase * 2j)

        return P

    def getOsc(self, alpha, beta):
        P = self.getAmps(alpha, beta)
        pOsc = np.abs(P)**2
        return pOsc

    # Function to update hamiltonian if any input parameters are changed
    def update(self):
        if len(self.masses) > 3:
            temp_masses = self.masses[3:]
        else:
            temp_masses = []
        if len(self.masses) > 2:
            self.applyNominalHierarchy()
        if len(temp_masses) > 0:
            self.masses.extend(temp_masses)

        self.setPMNS(self.generations, self.mixingPars)
        self.setVanillaHamiltonian()
        self.vEigenvals, self.vMixingMatrix = self.getOrderedEigenObjects(self.vHam, vacuum=True)
        self.setFullHamiltonian()
        #print(self.hamiltonian)
        #print(self.eigenOrder)
        self.eigenvals, self.mixingMatrix = self.getOrderedEigenObjects(self.hamiltonian)

    def set_gens(self, ngens):
        self.generations = ngens
        if len(self.masses) > 3:
            temp_masses = self.masses[3:]
        else:
            temp_masses = []
        if len(self.masses) > 2:
            self.applyNominalHierarchy()
        if len(temp_masses) > 0:
            self.masses.extend(temp_masses)
        self.setPMNS(self.generations, self.mixingPars)
        self.setVanillaHamiltonian()

    def update_hamiltonian(self, *args, **kwargs):

        self.eigenOrder = np.array(range(self.generations))
        self.hamArgs = args
        self.hamKwargs = kwargs
        #self.E = args[0]
        eig1, M1 = self.getOrderedEigenObjects(self.vHam)
        log_vals = np.logspace(start=0, stop=2, num=100, endpoint=True)
        for i in range(101):
            temp_args = np.array(self.hamArgs) * i / 100
            self.newHam = self.functHam(self.E, *temp_args, **kwargs)
            # print(temp_args)
            self.setFullHamiltonian()
            # We use the hungarian algorithm to find the correct eigenvalue ordering
            eig2, M2 = self.getOrderedEigenObjects(self.hamiltonian)  # new eigenvectors
            # calculate the difference matrix
            # Calculate the differences between columns of M1 and M2
            D = np.abs(M1[:, :, np.newaxis] - M2[:, np.newaxis, :])

            # Solve the linear sum assignment problem
            cost_matrix = np.sum(D, axis=0)
            # use the Hungarian algorithm to find the minimum cost matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # reorder the eigenvalues and matrix according to the matching. New matrix becomes old
            M1 = M2[:, col_ind]
            # save the new ordering
        self.eigenOrder = col_ind
        self.update()

    def __update_hamiltonian(self, *args, **kwargs):
        self.hamArgs = args
        self.hamKwargs = kwargs
        #self. E = args[0]
        self.newHam = self.functHam(*args, **kwargs)
        self.update()
        return self.eigenvals

    def new_funcHamiltonian(self, newHamiltonian, *args, **kwargs):
        self.functHam = newHamiltonian
        self.hamArgs = args
        self.hamKwargs = kwargs

    def applyNominalHierarchy(self):
        temp_masses = []
        if len(self.masses) > 3:
           temp_masses = self.masses[3:]
        if self.IH:
            self.masses = [np.sqrt(7.42 * 10 ** (-5)), np.sqrt(2.51 * 10 ** (-3)), 0]
        else:
            self.masses = [0, np.sqrt(7.42 * 10 ** (-5)), np.sqrt(2.51 * 10 ** (-3))]
        self.masses.extend(temp_masses)

# A function containing the usual matter hamiltonian for n generations and a given electron density
def matterHamiltonian(energy, density, ngens=3, earthCrust=False, neOverNa=False, electronDensity=False):

    # Take care of the units
    if earthCrust:
        G_f = 5.3948e-5
    elif neOverNa:
        G_f = 5.4489e-5
    elif electronDensity:
        G_f = 9.93e-2
    else:
        G_f = 1.166e-5

    #  nominal matter hamiltonian
    H = np.zeros((ngens, ngens))
    H[0, 0] = 2 * energy * density * G_f * np.sqrt(2)  # sqrt(2)*Fermi_constant*electron_number_density
    if ngens > 3:
        for i in range(3, ngens):
            H[i, i] = -2/3*H[0, 0]
    return H

# A function for recovering mixing angles from a given 3x3 mixing matrix
def extractMixingAngles(mixMatrix):
    #  get mixing angles from generic unitary 3x3 matrix

    mMatrix = np.abs(mixMatrix)
    th13 = np.arcsin(mMatrix[0, 2])

    if np.cos(th13) != 0:
        th12 = np.arccos(mMatrix[0, 0] / np.cos(th13))
        th23 = np.arccos(mMatrix[2, 2] / np.cos(th13))
    else:
        if mMatrix[1, 1] != 0:
            th12 = np.arctan(mMatrix[1, 0] / mMatrix[1, 1])
            th23 = np.arcos(mMatrix[1, 1] / np.cos(th12))
        else:
            th12 = np.arctan(mMatrix[2, 0] / mMatrix[2, 1])
            th23 = np.arcos(mMatrix[1, 0] / np.sin(th12))

    if th13 != 0 and th23 != 0 and th12 != 0 and np.cos(th12) != 0 and np.cos(th23) != 0:
        mod = mMatrix[1, 1]**2
        numerator = np.cos(th12)**2 * np.cos(th23)**2 + np.sin(th12)**2 * np.sin(th23)**2 * np.sin(th13) ** 2 - mod
        denominator = 2*np.cos(th12)*np.sin(th12)*np.cos(th23)*np.sin(th23)*np.sin(th13)
        dcp_abs = np.arccos(numerator / denominator)
        dcp = - dcp_abs * np.sign(mixMatrix[0, 2].imag)
    else:
        dcp = 0

    return np.array([np.sin(th12)**2, np.sin(th23)**2, np.sin(th13)**2, dcp])

# A recursive function for finding an "optimal" binning in the approximation to a varying matter potential
def split_range(func, max_change, start, end, start0, end0):
    # Calculate the maximum absolute value of the derivative within the range
    # We need to make some guesses about how many bins we want.
    # Getting the derivative at many points might slow down everytihng
    points = np.linspace(start, end, 100)
    derivative = np.gradient(func(points), points) * (end - start)
    max_derivative = np.amax(np.abs(derivative))


    # If the maximum derivative is less than or equal to max_change, return the range as a single bin
    # We hardcode a maximum of bins:
    if end-start < (end0-start0) / 2000:
        print('WARNING: potential is very steep. Minimum bin width reached.')
        return [(start, end)]
    if max_derivative <= max_change:
        return [(start, end)]
    # Otherwise, split the range into two sub-ranges and recursively split each sub-range
    else:
        mid = (start + end) / 2
        left_bins = split_range(func, max_change, start, mid, start0, end0)
        right_bins = split_range(func, max_change, mid, end, start0, end0)
        return left_bins + right_bins

class VaryingPotentialSolver():
    # A class for solving a path integral approximately by dividing into small
    # sections of constant potential
    def __init__(self, propagator, matterHam, ne_profile, l_start, l_end, delta_bin, const_binWidth=0, earthCrust=False, neOverNa=False, electronDensity=False, ngens=3):
        # propagator:     a hamiltonian propagator
        # matterHam:      function of the potential to be fed to the propagator
        # WARNING!!: WHEN INSTANCING THE PROPAGATOR, YOU MUST USE THIS FOR THE
        # NEW HAMILTONIAN FUNCTION!
        # ne_profile:      a function of electron number density to feed to the matter Hamiltonian
        # l_start, l_end: boundaries for the potential method
        # delta_bin:      the bins need not be the same width. Here we allow a maximum
        #                 variation of the potential per bin.
        # const_binwidth  If non_zero, set the bin width to a constant. Useful when functions are
        #                 very quickly varying

        self.propagator = propagator
        self.ne_profile = ne_profile
        self.matterH = matterHam
        self.bounds = np.array([l_start, l_end])
        self.delta_bin = delta_bin
        self.const_binwidth = const_binWidth

        self.earthCrust = earthCrust
        self.neOverNa = neOverNa
        self.electronDensity = electronDensity
        self.ngens = ngens

        self.setBinnedPotential()


    def setBinnedPotential(self):
        # Get the propagation lengths and constant matter potential values for our bins
        if self.const_binwidth != 0:
            print('constant bin width specified:')
            print('Electron density function will be evaluated at bin centre')
            print('The last bin might be shorter than the rest')
            nbins = int((self.bounds[1] - self.bounds[0]) / self.const_binwidth)

            # Check we have an appropriate number of bins
            if nbins > 0:
                print('There will be ' + str(nbins+1) + ' bins in our calculation')
            else:
                print('ERROR: Bin width is larger than given range. Please choose a smaller width')
                return -1

            # Get bin centres: - last bin is the remainder
            self.binCentres = (np.arange(nbins + 1) + 0.5) * self.const_binwidth
            self.binCentres[-1] = self.binCentres[-2] + 0.5 * (self.const_binwidth + self.bounds[1] - self.binCentres[-2])

            # Bin widths is trivial
            self.binWidths = np.ones(nbins + 1) * self.const_binwidth
            self.binWidths[-1] = (self.bounds[1] - self.bounds[0]) - self.const_binwidth*nbins


        else:
            binEdges = np.array(split_range(self.ne_profile, self.delta_bin, self.bounds[0], self.bounds[1], self.bounds[0], self.bounds[1]))
            self.binCentres = np.mean(binEdges, axis=1)
            self.binWidths = np.diff(binEdges, axis=1)[:, 0]

        # Finally, get value of potential at each bin:
        self.binned_ne_profile = np.vectorize(self.ne_profile)(self.binCentres)

    def __transition_helper(self, n_e, L):
        # This function calculates the transition matrix for a uniform potential
        #matterPotential = self.matterH(n_e, earthCrust=self.earthCrust, neOverNa=self.neOverNa, electronDensity=self.electronDensity, ngens=self.ngens)

        # create a copy of the propagator to parallelise and set values
        temp_propagator = copy.deepcopy(self.propagator)
        temp_propagator.L = L
        # temp_propagator.newHam = matterPotential
        temp_propagator.functHam = self.matterH
        temp_propagator.update_hamiltonian(n_e, earthCrust=self.earthCrust, neOverNa=self.neOverNa, electronDensity=self.electronDensity, ngens=self.ngens)
        size = temp_propagator.generations

        amplitudes = np.zeros((size, size), dtype=complex)
        for i in range(size):
            for j in range(size):
                amplitudes[i, j] = temp_propagator.getAmps(i, j)
        return(amplitudes)

    def transition_helper_wrapper(self, args):
        # Unpack the arguments and call the original function
        return self.__transition_helper(*args)

    def setTransitionAmplitude(self):
        # Use a helper function to calculate the transition amplitude
        # for each bin and then multiply together appropriately
        # uses multiprocessing
        with Pool(processes=len(self.binCentres)) as pool:
            args = [(self.binned_ne_profile[i], self.binWidths[i]) for i in range(len(self.binned_ne_profile))]
            results = pool.map(self.transition_helper_wrapper, args)

        amp_product = reduce(np.matmul, results)
        self.transitionAmplitude = amp_product.transpose()

    # getter function
    def getTransitionAmplitude(self):
        return self.transitionAmplitude

    def getProbs(self, alpha, beta):
        P = np.abs(self.transitionAmplitude[alpha, beta] * self.transitionAmplitude[alpha, beta].conjugate())
        return P


