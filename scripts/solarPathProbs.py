import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator

# Solar electron number density
def solar_density(r):
    


energy = sys.argv[1]

matterHam = customPropagator.matterHamiltonian
prop = customPropagator.HamiltonianPropagator(0, 1, energy)

