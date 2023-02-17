import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
from numba import jit, cuda
# Takes an energy spectrum file, a density, and a baseline and calculates the necessary shift
# in dcp to move to the degenerate mass hierarchy point.
# saves to file "savefile"
# -------------------------------------------------------------------------------------------------------------------------#

# SHOULD BE USED AS: python hierarchyChange_GPU spectrumFileName.txt electronDensity baseline savefile.txt (optional sindcp)

# -------------------------------------------------------------------------------------------------------------------------#
def load_spectrum(filename):
    data = np.loadtxt(filename, dtype=np.float, ndmin=2)
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


# function optimized to run on gpu. Calculates the shifts
@jit(target_backend='cuda')
def get_shifts(matterH, energies, weights, l, myRange, sinMode, npoints):
    # Calculates the necessary dcp(or sindcp) shifts for given energies
    # Assumes remaining parameters are in asimov A
    dcps = np.linspace(myRange[0], myRange[1], npoints)

    # Remove NaNs and normalise weighs
    weights[np.isnan(weights)] = 1.0
    normed_weights = weights / np.sum(weights)

    if sinMode:
        inputs = np.arcsin(dcps)
    else:
        inputs = dcps
    shifts_NH = np.zeros((npoints, 2))  # There are up to 2 degenerate points
    shifts_IH = np.zeros((npoints, 2))  # Same for shifting from the inverted hierarchy
    avg_shifts_NH = np.zeros((npoints, 2))
    avg_shifts_IH = np.zeros((npoints, 2))
    probabilities = np.zeros(4)

    for j in range(npoints):
        # Set propagators to invertse and normal hierarchies, nu and nubar
        prop_NH_nu = customPropagator.HamiltonianPropagator(matterH, l, energies[j])
        prop_IH_nu = customPropagator.HamiltonianPropagator(matterH, l, energies[j])
        prop_NH_nub = customPropagator.HamiltonianPropagator(matterH, l, energies[j])
        prop_IH_nub = customPropagator.HamiltonianPropagator(matterH, l, energies[j])

        prop_IH_nu.IH = True
        prop_IH_nub.IH = True
        prop_NH_nub.antinu = True
        prop_IH_nub.antinu = True

        # Turn into vector:
        props = [prop_NH_nu, prop_NH_nub, prop_IH_nu, prop_IH_nub]
        vals = np.zeros((npoints, 2))

        # Loop through dcp
        for i in range(npoints):
            for propagator in props:
                propagator.mixingPars[3] = inputs[i]

            for k in range(4):
                probabilities[k] = props[k].getOsc(1, 0)

            vals[i, 0] = probabilities[0] - probabilities[1]
            vals[i, 1] = probabilities[2] - probabilities[3]

        # find shifts:
        # Normal Hierarchy
        for i in range(npoints):
            differences = np.absolute(vals[:, 1] - vals[i, 0])
            index = differences.argmin()
            if differences[index] < 0.0001:
                shifts_NH[i] = dcps[index] - dcps[i]

                # Wrap if necessary
                if shifts_NH[i] > myRange[1]:
                    shifts_NH[i] -= 2 * myRange[1]
                elif shifts_NH[i] < myRange[0]:
                    shifts_NH[i] += 2 * myRange[1]
            else:
                shifts_NH[i] = np.nan  # Return nan if no shift exists

        # Inverse Hierarchy
        for i in range(npoints):
            differences = np.absolute(vals[:, 0] - vals[i, 1])
            index = differences.argmin()
            if differences[index] < 0.0001:
                shifts_IH[i] = dcps[index] - dcps[i]

                # Wrap if necessary
                if shifts_IH[i] > myRange[1]:
                    shifts_IH[i] -= 2 * myRange[1]
                elif shifts_IH[i] < myRange[0]:
                    shifts_IH[i] += 2 * myRange[1]
            else:
                shifts_IH[i] = np.nan  # Return nan if no shift exists

        # Finally, add to the average:
        avg_shifts_NH += shifts_NH * normed_weights[j]
        avg_shifts_IH += shifts_IH * normed_weights[j]

    return avg_shifts_NH, avg_shifts_IH


def main():
    sinMode = IsSin()
    e_filename = sys.argv[1]  # Array containing the energies and their respective weighs
    n_e = float(sys.argv[2])  # Electron number density
    l = float(sys.argv[3])  # Baseline
    savefile = sys.argv[4]

    energies, weights = load_spectrum(e_filename)

    # Check if we are running in sine mode:
    if len(sys.argv) == 6:
        sinMode.setSin(True)

    print('************************************************************')
    print('Spectrum file = ' + e_filename)
    print('Density = ' + str(n_e))
    print('Baseline = ' + str(l))
    print('Save file will be ' + savefile)
    print('Running in sin(dcp) = ' + str(sinMode.sinMode))
    print('************************************************************')

    if sinMode.sinMode:
        myRange = [-1 + 0.0001, 1 - 0.0001]
    else:
        myRange = [-np.pi + 0.0001, np.pi - 0.0001]

    # Number of points you want to calculate the shifts at
    npoints = 10 ** 2

    # Set the matter effect's contribution to the Hamiltonian
    matterH = customPropagator.matterHamiltonian(n_e, 3)

    # Get shifts (GPU function)
    normal_hierarchy, inverse_hierarchy = get_shifts(matterH,
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