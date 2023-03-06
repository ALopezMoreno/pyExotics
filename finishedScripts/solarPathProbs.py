import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
from tqdm import tqdm

# Propagate neutrino oscillations through a potential like that of the sun
# Use: python solarPathProbs.py config.cfg
# Current use: python solarPathProbs.py Emax Emin nBinsInPotential nPointsForAverage


# Solar electron number density
def solar_density(l):
    solar_radius = 696340
    density = 245*np.exp(-10.54*l/solar_radius)
    return density

def vacuum(l):
    return 0.0

def read_config(filename):
    eMax = 0
    eMin = 0
    nPoints = 0
    maxChange = 0
    savefile = ''
    avg = 0

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("MAX_E:"):
                eMax = line.strip()[6:]
            elif line.startswith("MIN_E:"):
                eMin = line.strip()[6:]
            elif line.startswith("NUMBER_OF_POINTS:"):
                nPoints = line.strip()[17:].strip()
            elif line.startswith("OUTPUT:"):
                savefile = line.strip()[7:].strip()
            elif line.startswith("MAX_POTENTIAL_CHANGE_PER_BIN:"):
                maxChange = line.strip()[29:].strip()
            elif line.startswith("AVERAGING_POINTS:"):
                avg = line.strip()[17:].strip()

    return float(eMax), float(eMin), int(nPoints), float(maxChange), int(avg), savefile



def main():
    # HARDCODED!!
    getVacuum = False

    energyMax, energyMin, energyBin, max_change, avg, savefile = read_config(sys.argv[1])
    print('************************************************************')
    print('Energy range = ' + str(10 ** energyMin) + '-' + str(10 ** energyMax) + ' MeV')
    print('Maximum change per bin in Potential = ' + str(max_change))
    print('Number of points in moving average = ' + str(avg))
    print('Save file will be ' + savefile)
    print('*************************************************************')

    # Make list of energies to loop through
    energies = np.logspace(energyMin, energyMax, energyBin)


    matterHam = customPropagator.matterHamiltonian
    prop = customPropagator.HamiltonianPropagator(0, 1, 1)
    ne_profile = solar_density
    solver = customPropagator.VaryingPotentialSolver(prop, matterHam,
                                                     ne_profile, 0, 696340,
                                                     max_change)

    print('Resulting bins in Potential = ' + str(len(solver.binCentres)))
    probs = np.zeros(energyBin)

    # calculate transition amplitude from the surface of the sun to infinity
    # this does not depend on energy
    if getVacuum == True:
        solver.propagator.newHam = np.zeros((3, 3))
        solver.propagator.E = 10**-3
        solver.propagator.update()
        eigenvalues = solver.propagator.eigenvals
        lovere = 10 ** 15
        lengths = lovere / np.linspace(0.8, 1.2, 10 ** 4)
        vacuum_amps = np.zeros((3, 3), dtype=complex)

        for j in lengths:
            solver.propagator.L = j
            for k in range(3):
                for g in range(3):
                    vacuum_amps[k, g] += solver.propagator.getAmps(k, g)

        vacuum_amp = vacuum_amps / len(lengths)
        print(vacuum_amp[0, 0])
        print(np.abs(vacuum_amp[0, 0] * vacuum_amp[0, 0].conjugate()))


    for i, E in tqdm(enumerate(energies*10**-3), total=len(energies)):
        if avg == 0 or avg == 1:
            ens = np.asarray([E])
        else:
            ens = np.random.uniform(E*0.9, E*1.1, avg)
        print(ens)
        temp_probs = 0
        for j, energy in ens:
            # Assign energies
            solver.propagator.E = energy
            # Calculate transition amplitude inside the sun
            solver.setTransitionAmplitude()
            temp_probs += solver.getProbs(0, 0)
        # Average over energy range
        probs[i] = temp_probs / len(ens)

    data = np.column_stack((energies, probs))
    np.savetxt(savefile, data, delimiter="\t", fmt='%.9f')



if __name__ == '__main__':
    main()
