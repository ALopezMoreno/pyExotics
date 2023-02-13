import numpy as np
from findiff import FinDiff
from scipy.sparse.linalg import inv, eigs
from scipy.sparse import csc_matrix, eye, diags

class TDSEsolver():
    #  A class for solving the time dependent schrodinger equation given an oscillator and a potential.
    #  It uses the HamiltonianPropagator, feeds in the varying potential and solves the schrodinger equation numerically.
    def __init__(self, hamiltonianObject, distanceDependentPotential, x_range, t_range, x_grid=100, t_grid=100):
        #  hamiltonianObject:      HamiltonianPropagator to help us build the nxn PMNS matrix
        #  timeDependentPotential: Parameter dependent function. Output a matrix the same dimension as the PMNS
        #  x,t range               2-tuple indicating the range of values for which to solve the TDSE
        #  a negative endpoint indicates that we want to propagate forever
        #  x,t grid:               number of points in the discretisation of the x and t dimensions


        self.H = hamiltonianObject
        self.V = distanceDependentPotential
        self.x_start = x_range[0], self.x_end = x_range[1]
        self.t_start = t_range[0], self.t_end = t_range[1]
        self.__checkRanges()
        self.x_grid = x_grid
        self.t_grid = t_grid
        self.__setGrid()


    # -- PUBLIC FUNCTIONS -- #
    def propagate(self, initial_state):
        #  Propagate the initial state through the TDSE
        self.__update()
        psi = initial_state
        evolution = np.empty((self.t_grid, len(psi)), dtype=object)
        for it, t in enumerate(self.ts):
            psi = self.U.dot(psi)
            psi[0] = psi[-1] = 0
            evolution[it] = psi
        return(evolution)            


    # -- GETTER/SETTER FUNCTIONS -- #
    def get_hamiltonianObject(self):
        return self.H

    def set_hamiltonianObject(self, hamiltonianObject):
        self.H = hamiltonianObject

    def get_distanceDependentPotential(self):
        return self.V

    def set_distanceDependentPotential(self, distanceDependentPotential):
        self.V = distanceDependentPotential
    def get_x_start(self):
        return self.x_start

    def get_x_end(self):
        return self.x_end

    def set_x_range(self, x_range):
        self.x_start = x_range[0], self.x_end = x_range[1]
        self.__checkRanges()

    def get_t_start(self):
        return self.t_start

    def get_t_end(self):
        return self.t_end

    def set_t_range(self, t_range):
        self.t_start = t_range[0], self.t_end = t_range[1]
        self.__checkRanges()

    def get_x_grid(self):
        return self.x_grid

    def set_x_grid(self, x_grid):
        self.x_grid = x_grid

    def get_t_grid(self):
        return self.t_grid

    def set_t_grid(self, t_grid):
        self.t_grid = t_grid
        self.__setGrid()

    # -- PRIVATE METHODS -- #
    def __checkRanges(self, x_range, t_range):
        #  Check the time and distance ranges are valid. negative number in distance makes it "effectively" infinite
        if len(x_range) != 2 or len(t_range) != 2:
            print("ERROR: range must be of the form [start, end]")
            exit()

        if self.x_end < 0:
            self.x_end = 10**10
        if self.x_end <= self.x_start or self.t_end <= self.t_start:
            print("ERROR: endpoints must be larger than start points")
            exit()

    def __setGrid(self):
        #  Build grid from inputs
        self.x  = np.linspace(self.x_start, self.x_end, self.x_grid)
        self.ts = np.linspace(self.t_start, self.t_end, self.t_grid)
        self.dt = self.ts[1] - self.ts[0]

    def __update(self):
        #  Method for updating the discretisation of hamiltonian and TISE solvers if the parameters have been updated.
        #  Do it before propagating!

        #  Set the discretised hamiltonian
        self.discrete_H = self.H + self.V(self.x)

        # Build the Crank-Nicholson propagator:
        I_plus = csc_matrix(eye(len(self.x)) + 1j * self.dt / 2. * self.discrete_H)
        I_minus = csc_matrix(eye(len(self.x)) - 1j * self.dt / 2. * self.discrete_H)
        self.U = inv(I_minus).dot(I_plus)




