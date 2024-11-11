import numpy as np
import sys
import sympy
from tqdm import tqdm
sys.path.append('../')
import matplotlib.pyplot as plt
from graphing import plotting
parula = plotting.parula_map
from vanilla import oscillatorBase
from nonUnitary import sterileOsc
import experiments
from LED import KKmodes

#L_near = 297
E = 0.6


def prop(a,b, prop, L, E):
    eig1 = 0
    eig2 = 7.42*10**(-5)
    eig3 = 7.42*10**(-5) + 2.494*10**(-3)
    mix = prop.mixM

    t1 = mix[a, 0]*mix[b, 0].conjugate() * np.exp(complex(0, 1)*L/E * 1.26693281*2 * eig1)
    t2 = mix[a, 1]*mix[b, 1].conjugate() * np.exp(complex(0, 1)*L/E * 1.26693281*2 * eig2)
    t3 = mix[a, 2]*mix[b, 2].conjugate() * np.exp(complex(0, 1)*L/E * 1.26693281*2 * eig3)

    amplitude = t1 + t2 + t3
    return amplitude * amplitude.conjugate()

def prop4_2(a,b, prop, L, E):
    eig1 = 0
    eig2 = 7.42*10**(-5)
    eig3 = 7.42*10**(-5) + 2.494*10**(-3)
    eig4 = 10**100

    mix = prop.mixM
    total = 0
    nonUcol = [complex(np.cos(prop.phi14), -np.sin(prop.phi14))*np.sin(prop.theta14),
               complex(np.cos(prop.phi24), -np.sin(prop.phi24))*np.cos(prop.theta14)*np.sin(prop.theta24),
               complex(np.cos(prop.phi34), -np.sin(prop.phi34))*np.cos(prop.theta14)*np.cos(prop.theta24)*np.sin(prop.theta34)]
    for i in range(100000):
        L2 = L + L*np.random.uniform(-1, 1)*0.05
        t1 = mix[a, 0]*mix[b, 0].conjugate() * np.exp(complex(0, 1)*L2/E * 1.26693281*2 * eig1)
        t2 = mix[a, 1]*mix[b, 1].conjugate() * np.exp(complex(0, 1)*L2/E * 1.26693281*2 * eig2)
        t3 = mix[a, 2]*mix[b, 2].conjugate() * np.exp(complex(0, 1)*L2/E * 1.26693281*2 * eig3)
        t4 = nonUcol[a] * nonUcol[b].conjugate() * np.exp(complex(0, 1)*L2/E * 1.26693281*2 * eig4)  # Add decoherent mixing

        amplitude = t1 + t2 + t3 + t4
        total += amplitude * amplitude.conjugate()
    return total/100000

def prop4(a,b, prop, L, E):
    eig1 = 0
    eig2 = 7.42*10**(-5)
    eig3 = 7.42*10**(-5) + 2.494*10**(-3)


    mix = prop.mixM
    nonUcol = [complex(np.cos(prop.phi14), -np.sin(prop.phi14))*np.sin(prop.theta14),
               complex(np.cos(prop.phi24), -np.sin(prop.phi24))*np.cos(prop.theta14)*np.sin(prop.theta24),
               complex(np.cos(prop.phi34), -np.sin(prop.phi34))*np.cos(prop.theta14)*np.cos(prop.theta24)*np.sin(prop.theta34)]

    t1 = mix[a, 0]*mix[b, 0].conjugate() * np.exp(complex(0, 1)*L/E * 1.26693281*2 * eig1)
    t2 = mix[a, 1]*mix[b, 1].conjugate() * np.exp(complex(0, 1)*L/E * 1.26693281*2 * eig2)
    t3 = mix[a, 2]*mix[b, 2].conjugate() * np.exp(complex(0, 1)*L/E * 1.26693281*2 * eig3)
    t4 = nonUcol[a] * nonUcol[b].conjugate() * 0.5  # Add decoherent mixing

    amplitude = t1 + t2 + t3 + t4
    return amplitude * amplitude.conjugate()

prop2 = oscillatorBase.Oscillator(1, E)
propagator = sterileOsc.oneSterile_oscillator(1, E)

# set values for non-unitary parameters
propagator.theta14 = np.arcsin(np.sqrt(0.1))
propagator.theta24 = np.arcsin(np.sqrt(0.2))
propagator.phi14   = +0.5

# update parameters in object
propagator.set_nonUnitary()
propagator.build_mixing_matrix()

n = 100
L_near = np.logspace(0, 3, n)
p_ee_near = np.zeros(len(L_near))
p_mumu_near = np.zeros(len(L_near))
p_tautau_near = np.zeros(len(L_near))

a = np.zeros(len(L_near))
b = np.zeros(len(L_near))
c = np.zeros(len(L_near))
ref = np.zeros(len(L_near))

for i in tqdm(range(n)):
    propagator.L = L_near[i]
    prop2.L = L_near[i]
# get survival oscProbs at the near detector
    p_ee_near[i]   =   propagator.getOsc(1, 0)
    p_mumu_near[i] =   propagator.getOsc(0, 1)
    p_tautau_near[i] = propagator.getOsc(0, 2)

    ref[i] = prop2.getOsc(0, 0)
    #p_extra = propagator.getOsc(0, 0)
    # I have 0.64000
    # THIS IS WITH MATTER EFFECT!!!

    #print(p_ee_near)
    #print(p_mumu_near)
    #print(p_tautau_near)
    #print(p_ee_near + p_mumu_near + p_tautau_near)
    #print()
    a[i] = prop4_2(1, 0, propagator, L_near[i], E)
    b[i] = prop4(1, 0, propagator, L_near[i], E)
    c[i] = prop(1, 0, propagator, L_near[i], E)

    #print(a)
    #print(b)
    #print(c)
    #print(a+b+c)
    #print()
    #print(propagator.mixM)
    #print(prop(0, 1, propagator, L_near, E))
    #print(prop(0, 1, propagator, L_near, E))
    #print(prop(2, 2, propagator, L_near, E))
    #print(prop(0, 0, propagator, L_near, E))

fig, ax = plt.subplots()

plotting.niceLinPlot(ax, L_near, a, logx=True, logy=False, color='gray')
plotting.niceLinPlot(ax, L_near, p_ee_near, logx=True, logy=False, color='blue')
plotting.niceLinPlot(ax, L_near, b, logx=True, logy=False, color='red')
plotting.niceLinPlot(ax, L_near, ref, logx=True, logy=False, color='green')
#ax.set_ylim(0, 1)
plt.show()
