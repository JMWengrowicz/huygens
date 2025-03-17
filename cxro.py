import tarfile as tar
import pandas as pd
import physical_constants as const

from matplotlib import pyplot as plt
import numpy as np
import os




def get_upper_case_indices(s):
    return [i for i, c in enumerate(s) if c.isupper()]


def wavelength2eV(lam):
    # this function converts between energy in eV to the wavelength in m of a photon.

    f = const.c / lam
    eV = f * const.h / const.eV2J
    return eV


def eV2wavelength(eV):
    # this function converts between energy in eV to the wavelength in m of a photon.

    f = const.eV2J*eV / const.h
    lam = const.c/f
    return lam


def filter(material, width=np.array([1]), photon=np.array([wavelength2eV(1e-9)]), photon_type='eV'):

    if isinstance(width, int) or isinstance(width, float):
        width = np.array([width])
    if isinstance(photon, int) or isinstance(photon, float):
        photon = np.array([photon])
    # complex_list = ['Si3N4']
    # complex_density_list = [3.44]  # g/cm^3
    # loc = 'U:/Plasma_Physics/Jonathan Wengrowicz/Python Modules/'
    # loc = 'C:/Users/yonatanwe/PycharmProjects/beta_version_modules/CXRO'
    loc = 'C:/Users/jonat/Documents/huygens/'
    width = width*1e-9
    # arc_name = loc + 'sf.tar.gz'
    # periodic_table = loc + 'periodic_table.csv'
    arc_name = 'sf.tar.gz'
    periodic_table = loc + 'periodic_table.csv'
    periodic_table = pd.read_csv(periodic_table, usecols=['Symbol', 'Density', 'AtomicMass'])
    arc = tar.open(loc + arc_name)
    idx = get_upper_case_indices(material)
    idx.append(len(material)+1)
    # if material in complex_list:
    #     density = complex_density_list(complex_list.index(material))
    #     for 
    # else:
    if width.size != (len(idx)-1) and (len(idx)-1) != 1:
        print('number of materials is not equal to number of widths')
        return
    if photon_type != 'eV':
        lam = photon
        photon = wavelength2eV(photon)
    else:
        lam = eV2wavelength(photon)

    sub_T = np.zeros([len(photon), len(idx)-1], dtype="complex_")
    for i in range(len(idx)-1):
        m = material[idx[i]:(idx[i+1])]
        if (len(idx) - 1) != 1:
            w = width[i]
        elif (len(idx) - 1) == 1:
            w = width
            sub_T = np.zeros([len(photon), len(width)], dtype="complex_")

        table_name = m.lower()+'.nff'
        table = pd.read_csv(arc.extractfile(table_name), sep='\t')
        f1 = table['E(eV)']  # due to bug with the importing
        eV = f1.index
        f1 = f1.values
        f1 = np.interp(photon, eV, f1)
        f2 = table['f1']     # due to bug with the importing
        f2 = f2.values
        f2 = np.interp(photon, eV, f2)

        # if (m == 'H' or m == 'N') and i != 0:
        #     atomic_number = periodic_table.Symbol == prev_m
        # else:
        #     atomic_number = periodic_table.Symbol == m

        atomic_number = periodic_table.Symbol == m
        density = periodic_table.Density[atomic_number]  # [g/cm^3]
        density = density.values
        atomic_mass = periodic_table.AtomicMass[atomic_number]
        atomic_mass = atomic_mass.values
        atomic_density = 1e6*density/(atomic_mass*const.amu)  # [A/m^3]
        n = 1-atomic_density*const.r_e*(lam**2)*(f1+1j*f2)/(2*np.pi)  #

        k = (const.eV2J*photon / const.hbar) / (const.c / n)  # k in medium
        k0 = (const.eV2J*photon / const.hbar) / (const.c / 1)  # k in vacuum

        delta_k = k - k0

        if (len(idx) - 1) == 1:
            return np.exp(-1j * delta_k * w)

        sub_T[:, i] = np.exp(-1j * delta_k * w)

        prev_m = m

    field_transmission = np.prod(sub_T, 1)
    return field_transmission



    # for i in range(len(filenames)):
    #     filename = filenames[i]
    #     # filename = 'ar.nff'
    #     print(filename)
    #     tab = pd.read_csv(arc.extractfile(filename), '\t')
    #     f2 = tab['f1']
    #     f1 = tab['E(eV)']
    #     # eV = tab['E(eV)']
    #
    #     # plt.plot(f1)
    #     # plt.show(block=True)
    #     # plt.plot(f2)
    #     # plt.show(block=True)


def Si3N4(width=np.array([1]), photon=np.array([wavelength2eV(1e-9)]), photon_type='eV'):

    ## should be specified
    material = 'SiN'
    atoms_in_unit = [3, 4]
    complex_density = 3.44  # g/cm^3

    ## can be imported from table (using atoms_in_unit):
    molecular_mass = 140.286  # Si*3+N*4


    molecular_density = 1e6*complex_density/(molecular_mass*const.amu)  # [M/m^3]

    width = width*1e-9
    # arc_name = loc + 'sf.tar.gz'
    # periodic_table = loc + 'periodic_table.csv'
    arc_name = 'sf.tar.gz'
    periodic_table = 'periodic_table.csv'
    periodic_table = pd.read_csv(periodic_table, usecols=['Symbol', 'Density', 'AtomicMass'])
    arc = tar.open(arc_name)
    idx = get_upper_case_indices(material)
    idx.append(len(material)+1)

    if photon_type != 'eV':
        lam = photon
        photon = wavelength2eV(photon)
    else:
        lam = eV2wavelength(photon)

    sub_T = np.zeros([len(photon), len(idx)-1], dtype="complex_")
    for i in range(len(idx)-1):
        m = material[idx[i]:(idx[i+1])]
        table_name = m.lower()+'.nff'
        table = pd.read_csv(arc.extractfile(table_name), sep='\t')
        f1 = table['E(eV)']  # due to bug with the importing
        eV = f1.index
        f1 = f1.values
        f1 = np.interp(photon, eV, f1)
        f2 = table['f1']     # due to bug with the importing
        f2 = f2.values
        f2 = np.interp(photon, eV, f2)

        # if (m == 'H' or m == 'N') and i != 0:
        #     atomic_number = periodic_table.Symbol == prev_m
        # else:
        #     atomic_number = periodic_table.Symbol == m

        atomic_number = periodic_table.Symbol == m
        # density = periodic_table.Density[atomic_number]  # [g/cm^3]
        # density = density.values
        atomic_mass = periodic_table.AtomicMass[atomic_number]
        atomic_mass = atomic_mass.values

        # atomic_density = 1e6*density/(atomic_mass*const.amu)  # [A/m^3]
        atomic_density = molecular_density*atoms_in_unit[i]
        n = 1-atomic_density*const.r_e*(lam**2)*(f1+1j*f2)/(2*np.pi)  #

        k = (const.eV2J*photon / const.hbar) / (const.c / n)  # k in medium
        k0 = (const.eV2J*photon / const.hbar) / (const.c / 1)  # k in vacuum

        delta_k = k - k0

        sub_T[:, i] = np.exp(-1j * delta_k * width)

        prev_m = m

    field_transmission = np.prod(sub_T, 1)
    return field_transmission




