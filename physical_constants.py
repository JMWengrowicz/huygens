from numpy import pi
import numpy as np

h = 6.62607e-34  # planck constant [J*s]
hbar = h/(2*pi)  # reduced planck constant
c = 299792458    # speed of light [m/s]
eV2J = 1.6021766e-19  # Joules per 1eV [1electron*1Volt/1J]
r_e = 2.8179403227e-15  # radius of electron [m]
amu = 1.66053904e-24  # atomic mass unit [g]‎
k_B = 1.380649e-23  # Boltzmann constant [J/K]


def blackbody(eV=np.linspace(10,3000,300), T=120):
    '''
    This funtion recieves vector of eV values of photons and the temperature T [eV] of the source, and returns its
    corresponding intensity in [J/eV m^2 s] while the horizontal axis should be in [ev]. Integral over the spectrum in
    eV is giving the intensity in [W/m^2]
    '''
    f = eV*eV2J/h
    return ((2*eV2J*f**3)/(c**2))*(1/(np.exp(h*f/(T*eV2J))-1))