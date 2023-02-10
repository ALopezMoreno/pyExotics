import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../')

from HamiltonianSolver import customPropagator
from graphing import plotting

def save_data(filename, rho, osc):
    data = np.column_stack((rho, osc))
    np.savetxt(filename, data, delimiter="\t", fmt='%.9f')

def matterHamiltonian(density, ngens):
    #  nominal matter hamiltonian
    H = np.zeros((ngens, ngens))
    H[0, 0] = density * 1.663787e-5 * np.sqrt(2)
    if ngens>3:
        for i in range(3, ngens):
            H[i, i] = -2*H[0, 0]
    return H

def extractMixingAngles(mixMatrix):
    #  get mixing angles from generic unitary 3x3 matrix
    #  (ignores complex phases)
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

    return np.array([th12, th23, th13])


#%%

def main():
    if len(sys.argv) != 4:
        print("arguments should be 1:Energy in MeV, 2:number of points, 3:output file")
        return -1

    h4 = matterHamiltonian(0, 4)
    h3 = matterHamiltonian(0, 3)

    E = float(sys.argv[1])
    npoints = int(sys.argv[2])
    output = sys.argv[3]
    print('E = ' + str(E) + ' MeV')
    print('will calculate oscillation at ' + str(npoints) +' points')
    print('save file will be ' + output)
    print('************************************************************')

    centralE = E * 10**-3

    prop = customPropagator.HamiltonianPropagator(h3, 295/0.6*centralE, centralE)
    prop.masses = [0, np.sqrt(7.42 * 10 ** (-5)), np.sqrt(2.51 * 10 ** (-3)), 10**3]
    prop.mixingPars = [np.arcsin(np.sqrt(0.307)), np.arcsin(np.sqrt(0.022)), np.arcsin(np.sqrt(0.561)),
                       np.arcsin(np.sqrt(0.)), np.arcsin(np.sqrt(0.)), np.arcsin(np.sqrt(0.)), -1.601, 0.0, 0.0]
    prop.generations = 4
    prop.new_hamiltonian(h4)
    prop.update()

    #  calculate matter effect for varying densities
    start = -3
    end = 6

    rho = np.logspace(start, end, npoints)
    probL = np.zeros(npoints)

    probs = np.zeros((npoints, 3))

    #prop.E = energies[k]*10**-3
    prop.E = centralE
    for i in range(npoints):
        h3 = matterHamiltonian(rho[i], 4)
        prop.new_hamiltonian(h3)
        p_osc = [prop.getOsc(0, 0), prop.getOsc(0, 1), prop.getOsc(1, 1)]
        for j in range(3):
            probs[i][j] = p_osc[j]

    save_data(output, rho, probs)

if __name__ == '__main__':
    main()