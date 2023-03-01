import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
from multiprocessing import Pool
import time

# Takes an energy spectrum file, a density, and a baseline and calculates the necessary shift
# in dcp to move to the degenerate mass hierarchy point.
# saves to file "savefile"
# -------------------------------------------------------------------------------------------------------------------------#

# SHOULD BE USED AS: python hierarchyChange_parallel.py configFile.cfg

# spectrumFileName.txt electronDensity baseline savefile.txt (optional sindcp)

# -------------------------------------------------------------------------------------------------------------------------#
def read_config(filename):
    baseline = 0
    n_e = 0
    e_filename = ''
    savefile = ''
    sin_mode = False
    mixing_parameters = np.array([0.307, 0.561, 0.0220, 7.53e-5, 2.494e-3, -1.601])

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("BASELINE:"):
                baseline = line.strip()[9:]
            elif line.startswith("E_DENSITY:"):
                n_e = line.strip()[10:]
            elif line.startswith("FLUXFILE:"):
                e_filename = line.strip()[9:]
            elif line.startswith("OUTPUT:"):
                savefile = line.strip()[7:]
            elif line.startswith("SIN_DCP:"):
                sin_mode = True if line.strip()[8:].strip().lower() == "true" else False
            elif line.startswith("MIXPARS:"):
                mixing_parameters = line.strip()[8:]

    print(mixing_parameters)
    return float(baseline), float(n_e), e_filename, savefile, sin_mode, np.array(mixing_parameters, dtype=float)
def get_shifts_helper_wrapper(args):
    # Unpack the arguments and call the original function
    return get_shifts_helper(*args)

def load_spectrum(filename):
    data = np.loadtxt(filename, dtype=float, ndmin=2)
    num_cols = data.shape[1]

    if num_cols == 2:
        energies = data[:, 0]
        weights = data[:, 1]
        if energies.size != weights.size:
            raise ValueError(
                "The spectrum file must have two columns indicating energies and weights, with the same length.")
    else:
        raise ValueError("The spectrum file must have two (or one) columns indicating energies and weights.")

    return energies, weights


class IsSin():
    # Just a checker to see if we want to run with dcp or sin(dcp)
    def __init__(self, isSin=False):
        self.sinMode = isSin

    def setSin(self, isSin):
        self.sinMode = isSin


# function optimized to run in parallel. Calculates the shifts
def get_shifts_parallel(matterH, th12, th23, th13, dcp, energies, weights, l, myRange, sinMode, npoints):
    # Calculates the necessary dcp(or sindcp) shifts for given energies
    # Assumes remaining parameters are in asimov A
    dcps = np.linspace(myRange[0], myRange[1], npoints)
    prop = customPropagator.HamiltonianPropagator(matterH, l, 1)
    prop.mixingPars = [th12, th13, th23, dcp]

    # Remove NaNs and normalise weighs
    weights[np.isnan(weights)] = 1.0
    normed_weights = weights / np.sum(weights)

    if sinMode:
        inputs = np.arcsin(dcps)
    else:
        inputs = dcps

    avg_shifts_NH = np.zeros((npoints))
    avg_shifts_IH = np.zeros((npoints))

    avg_probs_NH = np.zeros((npoints))
    avg_probs_IH = np.zeros((npoints))
    print("starting parallelism")
    with Pool() as pool:
        # Create a list of argument tuples
        args_list = [(prop, matterH, energies, weights, l, myRange, sinMode, npoints, j, inputs) for j in range(len(energies))]
        # Map the function to the arguments using the pool of worker processes
        results = pool.map(get_shifts_helper_wrapper, args_list)

        # Unpack the results
        for result in results:
            j, shifts_NH_j, shifts_IH_j = result

            # add to the average:
            avg_probs_NH += shifts_NH_j * normed_weights[j]
            avg_probs_IH += shifts_IH_j * normed_weights[j]

    # finally, get shifts
    # Normal Hierarchy
    for i in range(npoints):
        differences = np.absolute(avg_probs_IH - avg_probs_NH[i])
        index = differences.argmin()
        if differences[index] < 0.0001:
            avg_shifts_NH[i] = dcps[index] - dcps[i]

            # Wrap if necessary
            if avg_shifts_NH[i] > myRange[1]:
                avg_shifts_NH[i] -= 2 * myRange[1]
            elif avg_shifts_NH[i] < myRange[0]:
                avg_shifts_NH[i] += 2 * myRange[1]
        else:
            avg_shifts_NH[i] = np.nan  # Return nan if no shift exists

    # Inverse Hierarchy
    for i in range(npoints):
        differences = np.absolute(avg_probs_NH - avg_probs_IH[i])
        index = differences.argmin()
        if differences[index] < 0.0001:
            avg_shifts_IH[i] = dcps[index] - dcps[i]

            # Wrap if necessary
            if avg_shifts_IH[i] > myRange[1]:
                avg_shifts_IH[i] -= 2 * myRange[1]
            elif avg_shifts_IH[i] < myRange[0]:
                avg_shifts_IH[i] += 2 * myRange[1]
        else:
            avg_shifts_IH[i] = np.nan  # Return nan if no shift exists

    return avg_shifts_NH, avg_shifts_IH

def get_shifts_helper(propagator, matterH, energies, weights, l, myRange, sinMode, npoints, j, inputs):

    # work with a copy of the input object:
    base_prop = propagator
    base_prop.new_ham = matterH
    base_prop.L = l
    base_prop.E = energies[j]
    # Set propagators to inverse and normal hierarchies, nu and nubar
    prop_NH_nu = base_prop
    prop_IH_nu = base_prop
    prop_NH_nub = base_prop
    prop_IH_nub = base_prop

    prop_IH_nu.IH = True
    prop_IH_nub.IH = True
    prop_NH_nub.antinu = True
    prop_IH_nub.antinu = True

    # Turn into vector:
    props = [prop_NH_nu, prop_NH_nub, prop_IH_nu, prop_IH_nub]
    probabilities = np.zeros(4)

    vals = np.zeros((npoints, 2))
    start_time = time.time()
    # Loop through dcp
    for i in range(npoints):
        for propagator in props:
            propagator.mixingPars[3] = inputs[i]
            propagator.update()

        for k in range(4):
            probabilities[k] = props[k].getOsc(1, 0)

        vals[i, 0] = probabilities[0] - probabilities[1]
        vals[i, 1] = probabilities[2] - probabilities[3]
        if i % 10 == 0:
            print("iteration no. "+str(i)+". We have been running for " + str(int(time.time() - start_time)) + " seconds")

    return j, vals[: ,0], vals[:, 1]


def main():
    sinMode = IsSin()

    l, n_e, e_filename, savefile, sin_mode, pars = read_config(sys.argv[1])

    th12 = np.arcsin(np.sqrt(pars[0]))
    th23 = np.arcsin(np.sqrt(pars[1]))
    th13 = np.arcsin(np.sqrt(pars[2]))
    dm12 = pars[3]
    dm23 = pars[4]
    dcp = pars[5]

    energies, weights = load_spectrum(e_filename)

    # Check if we are running in sine mode:
    sinMode.setSin(sin_mode)

    print('************************************************************')
    print('Spectrum file = ' + e_filename)
    print('Density = ' + str(n_e))
    print('Baseline = ' + str(l))
    print('Save file will be ' + savefile)
    print('Running in sin(dcp) = ' + str(sinMode.sinMode))
    print('Mixing pars are ' + str(pars))
    print('Number of Energy Bins: ' + str(len(energies)))
    print('We are not considering changes in the mass differences here')
    print('************************************************************')

    if sinMode.sinMode:
        myRange = np.array([-1 + 0.0001, 1 - 0.0001])
    else:
        myRange = np.array([-np.pi + 0.0001, np.pi - 0.0001])

    # Number of points you want to calculate the shifts at
    npoints = 100 # look for prime numbers and combine?

    # Set the matter effect's contribution to the Hamiltonian
    matterH = customPropagator.matterHamiltonian(n_e, 3)

    # Get shifts (GPU function)
    normal_hierarchy, inverse_hierarchy = get_shifts_parallel(matterH,
                                                     th12, th23, th13, dcp,
                                                     energies,
                                                     weights,
                                                     l,
                                                     myRange,
                                                     sinMode.sinMode,
                                                     npoints)

    dcps = np.linspace(myRange[0], myRange[1], npoints)
    data = np.column_stack((dcps, normal_hierarchy, inverse_hierarchy))
    np.savetxt(savefile, data, delimiter="\t", fmt='%.9f')


if __name__ == '__main__':
    main()
