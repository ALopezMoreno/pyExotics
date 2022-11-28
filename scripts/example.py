# this is an example file for using this module

# Some useful imports
import numpy as np
import sys
import sympy
from tqdm import tqdm
sys.path.append('../')
import matplotlib.pyplot as plt
from graphing import plotting
parula = plotting.parula_map
from vanilla import oscillatorBase
from nonUnitary import Matter_potential
import experiments
from LED import KKmodes


# first, instance an experiment with a predicted flux in the near detector:

L = 295.2
Lnear = 0.280
profile='T2K'

# here the smoothing corresponds to taking the average oscillation probability from (100) uniformly sampled
# values in the energy bin

# Note that I am instancing the energy profile in the instance of the experiment (naughty naughty!)
exp = experiments.experiment(experiments.eProfile(profile), L, smooth=100)

# Instance a propagator. In our case, we will have two. The energy at instancing does not matter because we
# will re-write it - We are not specifying the ordering so Normal is taken by default:

# some parameters relevant to the LED model
mDirac = 10**-2
R = 10**-1
propLED = KKmodes.KKoscillator(L, 1, KKmodes.KKtower(10, mDirac, R, inverted=False, approx=True))

# Nominal 3-flavour propagator
propVanilla = oscillatorBase.Oscillator(L, 1)

# We now get the spectra for each mu-e channel in the far detector. Here we specify nu_bar modes in the
# appearance channel:
p_ee_LED = exp.propagate(propLED, 0, 0)
p_mue_LED = exp.propagate(propLED, 1, 0)
p_mue_bar_LED = exp.propagate(propLED, 1, 0, antineutrino=True)
p_mumu_LED = exp.propagate(propLED, 1, 1)

p_ee_Vanilla = exp.propagate(propVanilla, 0, 0)
p_mue_Vanilla = exp.propagate(propVanilla, 1, 0)
p_mue_bar_Vanilla = exp.propagate(propVanilla, 1, 0, antineutrino=True)
p_mumu_Vanilla = exp.propagate(propVanilla, 1, 1)

# We may now do whatever we want with these!!!