import numpy as np
import sympy as sym
import textwrap

th12, th23, th13, E = sym.symbols('th12, th23 th13, E', real=True)
n_e, Gf, dm21, dm31, dm41, gamma, m1, m2, m3 = sym.symbols('n_e Gf dm21 dm31 dm41 gamma m1 m2 m3', real=True)

e1, e2, e3, mu1, mu2, mu3, tau1, tau2, tau3 = sym.symbols('e1 e2 e3 mu1 mu2 mu3 tau1 tau2 tau3', real=True)

""""
A = sym.Matrix([[1, 0, 0], [0, sym.cos(th23), sym.sin(th23)], [0, -sym.sin(th23), sym.cos(th23)]])
B = sym.Matrix([[sym.cos(th13), 0, sym.sin(th13)], [0, 1, 0], [-sym.sin(th13), 0, sym.cos(th13)]])
C = sym.Matrix([[sym.cos(th12), sym.sin(th12), 0], [-sym.sin(th12), sym.cos(th12), 0], [0, 0, 1]])
#Simplyfications: Gf * n_e = 13, dm21 = 7, dm31 = 5

U2 = sym.Matrix([[e1, e2, e3], [mu1, mu2, mu3], [tau1, tau2, tau3]])

U = A*B*C

D = sym.Matrix([[m1**2, 0, 0], [0, m2**2, 0], [0, 0, m3**2]])
V = sym.Matrix([[gamma, 0, 0], [0, 0, 0], [0, 0, 0]])

M = U*D*U.H #+ V
print(textwrap.fill(str(M), width=160, replace_whitespace=False))

P, D_diag = M.diagonalize()

print("new mixing matrix:")
print(textwrap.fill(str(P), width=160, replace_whitespace=False))

eigenvals = M.eigenvals()
print("Eigenvalues:")
print(textwrap.fill(str(eigenvals), width=160, replace_whitespace=False))
"""

U = sym.Matrix([[sym.cos(th12), sym.sin(th12)], [-sym.sin(th12), sym.cos(th12)]])
V = sym.Matrix([[Gf, 0], [0, 0]])
D = sym.Matrix([[m1**2/(2*E), 0], [0, m2**2/(2*E)]])

M = U*D*U.H + V

M2 = sym.Matrix([[2*E*Gf + m1**2*sym.cos(th12)**2 + m2**2*sym.sin(th12)**2, (m2**2 - m1**2)/2*sym.sin(2*th12)],
                 [(m2**2 - m1**2)/2*sym.sin(2*th12), m2**2*sym.cos(th12)**2 + m1**2*sym.sin(th12)**2]])

#P, D_diag = M2.diagonalize()

#print(M)
eigenvals = M.eigenvals()
print("Eigenvalues:")

#print("new mixing matrix:")

print(textwrap.fill(str(eigenvals), width=160, replace_whitespace=False))
#print(" ")
#print(textwrap.fill(str(D_diag), width=160, replace_whitespace=False))