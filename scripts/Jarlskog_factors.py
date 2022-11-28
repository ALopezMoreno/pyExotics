import numpy as np
import sys
sys.path.append('../')
from graphing import plotting
from nonUnitary import sterileOsc
from vanilla import oscillatorBase
from vanilla import unitarityTests as uTest
import matplotlib.pyplot as plt
from tqdm import tqdm

def gaussian(x, mu, sig):
    chi = (x - mu) / sig
    gauss = np.exp(-0.5 * chi * chi)
    return gauss

throws =  10 ** 6


S12 = np.random.uniform(0, 1., throws)
S23 = np.random.uniform(0, 1., throws)
S13 = np.random.uniform(0, 0.002, throws)
dcp = np.random.uniform(-np.pi, np.pi, throws)

S14 = np.random.uniform(0, 0.08, throws)# np.zeros(throws) #np.random.uniform(0, 1., throws)
S24 = 1 - np.sqrt(np.random.uniform(0, 1., throws))#np.zeros(throws) #np.random.uniform(0, 1., throws)
#S34 = np.random.uniform(0, 1., throws)#np.zeros(throws) #np.random.uniform(0, 1., throws)
p14 = np.random.uniform(-np.pi, np.pi, throws)# np.zeros(throws) #np.random.uniform(-np.pi, np.pi, throws)
p24 = np.random.uniform(-np.pi, np.pi, throws)#np.zeros(throws) #np.random.uniform(-np.pi, np.pi, throws)
#p34 = np.random.uniform(-np.pi, np.pi, throws)#np.zeros(throws) #np.random.uniform(-np.pi, np.pi, throws)


print('loading posterior')
#S12, S23, S13, dcp, S14, S24, S34, p14, p24, p34, logl, acceptance, steptime, nstep = np.loadtxt(sys.argv[1], skiprows=1, max_rows=10 ** 6, dtype=float, delimiter=",").T
#throws = len(S12) #10 ** 6  # Number of throws according to the Haar measure
#print('posterior loaded')
#S14 = np.zeros(throws)
#S24 = np.zeros(throws)
#S34 = np.zeros(throws)
#p14 = np.zeros(throws)
#p24 = np.zeros(throws)
#p34 = np.zeros(throws)

# APPLY SOLAR AND REACTOR CONSTRAINTS ON 3+1 PICTURE
w_solar = np.ones(throws)
w_solar_vanilla = gaussian(S12, 0.307, np.sqrt(0.001689))
w_reactor_vanilla = gaussian(S13, 0.02, 0.0007)
# osc = oscillatorBase.Oscillator(1, 1)
osc = sterileOsc.oneSterile_oscillator(1, 1)

calc = uTest.Jfactors(osc.PMNS)
factorList = np.zeros((throws, 3, 3))
for i in tqdm(range(throws)):
    osc.theta12 = np.arcsin(np.sqrt(S12[i]))
    osc.theta23 = np.arcsin(np.sqrt(S23[i]))
    osc.theta13 = np.arcsin(np.sqrt(S13[i]))

    osc.theta14 = np.arcsin(np.sqrt(S14[i]))
    osc.theta24 = np.arcsin(np.sqrt(S24[i]))
    #osc.theta34 = np.arcsin(np.sqrt(S34[i]))

    osc.dcp = dcp[i]
    osc.p14 = p14[i]
    osc.p24 = p24[i]
    #osc.p34 = p34[i]

    osc.setPMNS()
    osc.set_nonUnitary()
    osc.build_mixing_matrix()
    calc.update(osc.mixM)
    w_solar[i] = gaussian(osc.A[0, 0]**4 * S12[i], 0.307, np.sqrt(0.001689))


    factorList[i] = calc.factors
ws_total = np.multiply(w_solar_vanilla, w_reactor_vanilla)

#plotting.plotPMNS(factorList, abs=False, Jpreset=True, lowlim=-0.04, highlim=0.04, weights=ws_total, bins=40)
#plt.suptitle("Haar U3 nominal Jarlskog factor priors + solar + reactor", fontsize=12)

#plt.show()
fig, ax = plt.subplots()

ax.hist(S12, bins=60, color='red', density=True, weights=w_solar_vanilla)
ax.hist(S12, bins=60, weights=w_solar, color='blue', density=True, alpha = 0.6)
ax.set_xlabel('sin2th12')
plt.suptitle("solar+reactor constraint in 3+0 (red) and 3+1 (blue) scenarios", fontsize=12)
plt.savefig('../images/s12_nominal_solar_reactor_v2.png')

plt.show()
