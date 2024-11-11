import HNLpropagatorTest
import numpy as np

a11 = 0.3
a22 = 0.3
a33 = 0.3

a21 = 0.2
a31 = 0.2
a32 = 0.2

mixPars = [np.arcsin(np.sqrt(0.307)),
           np.arcsin(np.sqrt(0.022)),
           np.arcsin(np.sqrt(0.561)),
           -1.601,
           a11, a21, a22, a31, a32, a33]


prop = HNLpropagatorTest.HNLpropagator(mixPars, 0, 0)
prop.update()

M = prop.mixingMatrix

print(np.abs(M[0, 0])**2 + np.abs(M[0, 1])**2 + np.abs(M[0, 2])**2)
