import sys
sys.path.append('../')
import numpy as np
from HamiltonianSolver import customPropagator
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from graphing import plotting
from itertools import permutations
import matplotlib.ticker as ticker
import experiments

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})


file_path = "../oscillationProbs/output_vanilla.txt"
data = np.loadtxt(file_path)

x = data[:, 0]
y = data[:, 1]

fig, ax = plt.subplots(dpi=300)

plotting.niceLinPlot(ax, x, y, logx=True, logy=False, color='black', linewidth=1)
plt.show()