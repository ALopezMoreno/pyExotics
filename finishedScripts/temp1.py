import numpy as np

def solar_density(l):
    solar_radius = 1
    density = 245*np.exp(-10.54*l/solar_radius)
    return density

print(solar_density(0))
print(solar_density(0.05))
print(solar_density(0.06))
print(solar_density(0.08))
print(solar_density(0.10))