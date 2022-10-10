# Class for calculating unitarity triangles (and Jarlskog factors) of a 3x3 mixing matrix
import numpy as np


class UnitarityGeometries:
    # There are 6 possible "triangles", which are only triangles if the matrix is unitary, so we save them as cuadrilaterals
    # There are 3 possible "quadrilaterals, which become pentagons if the matrix is not unitary
    # The triangles are related to row-column operations
    def __init__(self, mixingMatrix, majorana_phase_1, majorana_phase_2):
        self.e_mu = np.zeros(4, dtype=complex)
        self.e_tau = np.zeros(4, dtype=complex)
        self.mu_tau = np.zeros(4, dtype=complex)

        self.one_two = np.zeros(4, dtype=complex)
        self.one_three = np.zeros(4, dtype=complex)
        self.two_three = np.zeros(4, dtype=complex)

        self.diagOne = np.zeros(5, dtype=complex)
        self.diagTwo = np.zeros(5, dtype=complex)
        self.diagThree = np.zeros(5, dtype=complex)

        self.matrix = mixingMatrix

        self.r = majorana_phase_1
        self.s = majorana_phase_2  # THESE ARE YET TO BE IMPLEMENTED
        self.get_shapes()

    # Calculate the vertices. With normalisations as per "Leptonic Unitarity Triangles"
    # This forces two of the vertices to be (0, 0) and (0, 1), which is v nice, but does not allow for Maj. phases
    def get_shapes(self):
        # Row triangles
        self.e_mu[1] = complex(1, 0)
        self.e_mu[2] = self.e_mu[1] + (self.matrix[1, 1] * self.matrix[0, 1].conjugate()) / (
                    self.matrix[1, 0] * self.matrix[0, 0].conjugate())
        self.e_mu[3] = self.e_mu[2] + (self.matrix[1, 2] * self.matrix[0, 2].conjugate()) / (
                    self.matrix[1, 0] * self.matrix[0, 0].conjugate())

        self.e_tau[1] = complex(1, 0)
        self.e_tau[2] = self.e_tau[1] + (self.matrix[2, 1] * self.matrix[0, 1].conjugate()) / (
                    self.matrix[2, 0] * self.matrix[0, 0].conjugate())
        self.e_tau[3] = self.e_tau[2] + (self.matrix[2, 2] * self.matrix[0, 2].conjugate()) / (
                    self.matrix[2, 0] * self.matrix[0, 0].conjugate())

        self.mu_tau[1] = complex(1, 0)
        self.mu_tau[2] = self.mu_tau[1] + (self.matrix[2, 1] * self.matrix[1, 1].conjugate()) / (
                    self.matrix[2, 0] * self.matrix[1, 0].conjugate())
        self.mu_tau[3] = self.mu_tau[2] + (self.matrix[2, 2] * self.matrix[1, 2].conjugate()) / (
                    self.matrix[2, 0] * self.matrix[1, 0].conjugate())

        # Column triangles
        self.one_two[1] = complex(1, 0)
        self.one_two[2] = self.one_two[1] + (self.matrix[1, 0] * self.matrix[1, 1].conjugate()) / (
                    self.matrix[0, 0] * self.matrix[0, 1].conjugate())
        self.one_two[3] = self.one_two[2] + (self.matrix[2, 0] * self.matrix[2, 1].conjugate()) / (
                    self.matrix[0, 0] * self.matrix[0, 1].conjugate())

        self.one_three[1] = complex(1, 0)
        self.one_three[2] = self.one_three[1] + (self.matrix[1, 0] * self.matrix[1, 2].conjugate()) / (
                    self.matrix[0, 0] * self.matrix[0, 2].conjugate())
        self.one_three[3] = self.one_three[2] + (self.matrix[2, 0] * self.matrix[2, 2].conjugate()) / (
                    self.matrix[0, 0] * self.matrix[0, 2].conjugate())
        
        self.two_three[1] = complex(1, 0)
        self.two_three[2] = self.two_three[1] + (self.matrix[1, 1] * self.matrix[1, 2].conjugate()) / (
                    self.matrix[0, 1] * self.matrix[0, 2].conjugate())
        self.two_three[3] = self.two_three[2] + (self.matrix[2, 1] * self.matrix[2, 2].conjugate()) / (
                    self.matrix[0, 1] * self.matrix[0, 2].conjugate())
        
        self.diagOne[1] = complex(-1, 0)
        self.diagOne[2] = self.diagOne[1] + self.matrix[0, 0] * self.matrix[0, 0].conjugate()
        self.diagOne[3] = self.diagOne[2] + self.matrix[0, 1] * self.matrix[0, 1].conjugate()
        self.diagOne[4] = self.diagOne[3] + self.matrix[0, 2] * self.matrix[0, 2].conjugate()

        self.diagTwo[1] = complex(-1, 0)
        self.diagTwo[2] = self.diagTwo[1] + self.matrix[1, 0] * self.matrix[1, 0].conjugate()
        self.diagTwo[3] = self.diagTwo[2] + self.matrix[1, 1] * self.matrix[1, 1].conjugate()
        self.diagTwo[4] = self.diagTwo[3] + self.matrix[1, 2] * self.matrix[1, 2].conjugate()

        self.diagThree[1] = complex(-1, 0)
        self.diagThree[2] = self.diagThree[1] + self.matrix[2, 0] * self.matrix[2, 0].conjugate()
        self.diagThree[3] = self.diagThree[2] + self.matrix[2, 1] * self.matrix[2, 1].conjugate()
        self.diagThree[4] = self.diagThree[3] + self.matrix[2, 2] * self.matrix[2, 2].conjugate()


class Jfactors:
    # Calculate all nine Jarlskog factors for a given mixing matrix
    def __init__(self, mixing_matrix):
        self.Mmatrix = mixing_matrix
        self.factors = np.zeros((3, 3), dtype='double')
        self.get_factors()

    def get_factors(self):
        self.factors[0, 0] = (self.Mmatrix[1, 1] * self.Mmatrix[2, 2] *
                              self.Mmatrix[1, 2].conjugate() * self.Mmatrix[2, 1].conjugate()).imag
        self.factors[0, 1] =-(self.Mmatrix[1, 0] * self.Mmatrix[2, 2] *
                              self.Mmatrix[1, 2].conjugate() * self.Mmatrix[2, 0].conjugate()).imag
        self.factors[0, 2] = (self.Mmatrix[1, 0] * self.Mmatrix[2, 1] *
                              self.Mmatrix[1, 1].conjugate() * self.Mmatrix[2, 0].conjugate()).imag

        self.factors[1, 0] =-(self.Mmatrix[0, 1] * self.Mmatrix[2, 2] *
                              self.Mmatrix[0, 2].conjugate() * self.Mmatrix[2, 1].conjugate()).imag
        self.factors[1, 1] = (self.Mmatrix[0, 0] * self.Mmatrix[2, 2] *
                              self.Mmatrix[0, 2].conjugate() * self.Mmatrix[2, 0].conjugate()).imag
        self.factors[1, 2] =-(self.Mmatrix[0, 0] * self.Mmatrix[2, 1] *
                              self.Mmatrix[0, 1].conjugate() * self.Mmatrix[2, 0].conjugate()).imag

        self.factors[2, 0] = (self.Mmatrix[0, 1] * self.Mmatrix[1, 2] *
                              self.Mmatrix[0, 2].conjugate() * self.Mmatrix[1, 1].conjugate()).imag
        self.factors[2, 1] =-(self.Mmatrix[0, 0] * self.Mmatrix[1, 2] *
                              self.Mmatrix[0, 2].conjugate() * self.Mmatrix[1, 0].conjugate()).imag
        self.factors[2, 2] = (self.Mmatrix[0, 0] * self.Mmatrix[1, 1] *
                              self.Mmatrix[0, 1].conjugate() * self.Mmatrix[1, 0].conjugate()).imag

    def update(self, mixing_matrix):
        self.Mmatrix = mixing_matrix
        self.get_factors()