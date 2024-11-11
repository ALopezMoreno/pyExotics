import sys
sys.path.append('../')
sys.path.append('./')
import numpy as np
import experiments

t2kFlux = experiments.eProfile('T2K')

energies = np.zeros(len(t2kFlux.numu))
for i, binEdge in enumerate(t2kFlux.numuBE[1:]):
    energies[i] = t2kFlux.numuBE[i-1] + 0.5*(t2kFlux.numuBE[i] - t2kFlux.numuBE[i-1])

data = np.column_stack((energies, t2kFlux.numu))
np.savetxt('fluxFiles/t2kFlux.txt', data, delimiter="\t", fmt='%.9f')