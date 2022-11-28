# A class for playing around with 3+1 matter potential scenarios
# We take the sterile element of the mass matrix to be O(M_4*Enu)
import numpy as np

class OscHamiltonian:
    def __init__(self, M4, Enu, mass_scale, N_e, mixing_scale, sterileMatter=True):
        # inputs are mass of sterile, neutrino energy, mass of the lightest neutrino, matter electron density and
        # scale of the mixing term
        # normal ordering is assumed. N_e approx 2.6

        m1 = mass_scale
        m2 = m1 + np.sqrt(7.53e-5)
        m3 = m1 + np.sqrt(2.494e-3)

        self.sterileMatter = sterileMatter
        self.M_sterile = M4
        self.central_energy = Enu
        self.N_e = N_e
        self.nonUmix = mixing_scale
        # Mass matrix
        self.M_nu = np.zeros((4, 4))
        self.M_nu[0, 0] = m1*m1
        self.M_nu[1, 1] = m2*m2
        self.M_nu[2, 2] = m3*m3
        self.M_nu[3, 3] = self.M_sterile*self.central_energy

        # Matter potential
        temp1 = np.sqrt(2) * 1.663787*10**(-5) * self.N_e
        self.V_nu = np.zeros((4, 4))
        self.V_nu[0, 0] = 2*temp1
        if self.sterileMatter:
            self.V_nu[3, 3] = -1*temp1

        # Mixing matrix
        self.U = np.zeros((4, 4), dtype=complex)
        self.set_nonU_part()
        self.set_hamiltonian()

    def reset_mixing_matrix(self):
        # Set the 3x3 subsection of the PMNS to the nominal values:
        self.U[0, 0] = complex(0.82325816, 0)
        self.U[0, 1] = complex(0.54794708, 0)
        self.U[0, 2] = complex(-.00447925, 0.14825632)

        self.U[1, 0] = complex(-.36432154, 0.09244028)
        self.U[1, 1] = complex(0.55342666, 0.06152673)
        self.U[1, 2] = complex(0.74071452, 0)

        self.U[2, 0] = complex(0.41747302, 0.08177341)
        self.U[2, 1] = complex(-.62187224, 0.05442703)
        self.U[2, 2] = complex(0.65524194, 0)

        # Clear nonU part

        for i in range(3):
            self.U[3, i] = complex(0, 0)
            self.U[i, 3] = complex(0, 0)
        self.U[3, 3] = 1
    def set_nonU_part(self):
        # We assume the off-diagonal elements of the 4th col/row are all the same, and apply a global correction:
        # We assume no CPV in HNL sector

        self.reset_mixing_matrix()
        prod = np.sqrt(1 - self.nonUmix ** 2)
        self.U = np.multiply(self.U, prod)

        for i in range(3):
            self.U[3, i] = complex(self.nonUmix, 0)
            self.U[i, 3] = complex(self.nonUmix, 0)
        self.U[3, 3] = complex(np.sqrt(1 - 3 * self.nonUmix ** 2), 0)


    def set_hamiltonian(self):
        # Get the Hamiltonian!
        U_dagger = (self.U.conjugate()).transpose()
        prod1 = np.matmul(self.U, self.M_nu)

        self.H = 1 / (2 * self.central_energy) * np.matmul(prod1, U_dagger) + self.V_nu