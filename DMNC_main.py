# Imports************************************************************
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import time

import DMNC_Detector as dmnc_det     # Comes with plt, np, rand, time


# Less safe import statement, because DMNC_Rates is currently global
#      namespace, and is not contained within a class.
from DMNC_Rates import *    # Access to many fundamental calculations

'''Warning: This will bring in all the free parameters and functions
            from DMNC_Rates without ANY namespace protection. In
            particular, DO NOT overwrite the following variables:
            
            Z
            A
            e
            mu
            V0
            k
            R
            levels
            
            Note that the functions defined in DMNC_Rates.py will
            also exist in the global namespace. They have rather
            specific names, but do double check you don't overwrite
            anything while using these files.
            
            This can be made safer by eventually giving DMNC_Rates.py
            a class inside of it, but this will take a rather long
            amount of time, given how many "self." typings I'll need
            to add to the file.'''

# Functions**********************************************************

def sum_dict_vals(dictionary):
    '''Finds and returns the sum of the passed dictionary's values.
    
       Returns: the sum of a dictionary's stored data'''
    
    total = 0
    for key in dictionary.keys():
        total += dictionary[key]
    return total


def key_val_by_weight(dictionary):
    '''Selects a key-value pair from the dictionary by weight.
       The weighting is determined from the values, not keys.
    
       Returns: Tuple containing a selected key and value'''
    
    keys = list(dictionary.keys())
    total = sum_dict_vals(dictionary)
    weight = rand.uniform(0, total)
    
    # Define outside loop so they're in the correct namespace
    count = 0
    curr_key = 0
    curr_val = 0
    for i in range(len(keys)):
        curr_key = keys[i]
        curr_val = dictionary[curr_key]
        count += curr_val
        if count >= weight:
            return (curr_key, curr_val)
    # Temporary statement for testing purposes
    print('ERROR: "key_val_by_weight()" Exited the loop somehow')
    return (curr_key, curr_val)
    

def format_seconds(seconds):
    '''Takes a number of seconds and converts it to
       hour : minute : second format.
       
       Returns: string of hours:minutes:seconds'''
    
    hours = int(seconds // 3600)
    minutes = int(seconds // 60) % 60
    seconds = seconds % 60
    return f'{hours}:{minutes}:{seconds:.3f}'

# Parameters*********************************************************
face_count = {'front': 0,
              'back': 0,
              'right': 0,
              'left': 0,
              'top': 0,
              'bottom': 0}

det_length = 62                        # meters
det_width = 15.1                       # meters
det_height = 14                        # meters

# FIXME: R value from Rates is used, not this one. I think I'll have
# to turn Rates into a class so I can modify values.
R = 10                                 # DM radius, GeV^-1
x_sec_dict = xsec_v_tot_S()            # keys: states, vals: X sects
xsec_tot = sum_dict_vals(x_sec_dict)   # Total cross section, GeV^-2

# hc is 1240 eV * nm: e-9 for eV -> GeV, e-7 for nm -> cm
cm_from_inv_gev = 1240 / (2 * np.pi) * 1e-16

xsec_cm = xsec_tot * cm_from_inv_gev**2  # Total cross section, cm^2

# Number density of Liquid Argon, cm^-3:
num_density_LAr = 1.39 * 6.02e23 / 39.948

# Main***************************************************************

# Right now I am using this section for miscellaneous tests.
# I comment out the ones not in use, but save them because
# they are still useful for seeing how things are working.

det = dmnc_det.Detector(det_length, det_width, det_height, num_density_LAr, xsec_cm, x_sec_dict)

'''
for i in range(500000):
    face_count[det.random_face()] += 1

plt.title("Histogram of Entrance Location")
plt.grid(True)
plt.xlim(0, 7)
plt.bar(x=[i for i in range(1, 7)], height=[face_count[key] for key in face_count.keys()], tick_label = list(face_count.keys()))
plt.show()
'''

'''
for i in range(1000):
    det.random_entrance()
    if not det.particle_in_det():
        print('Test failed: particle outside detector')
        break
    unit_norm = np.sqrt(det.ux**2 + det.uy**2 + det.uz**2)
    if unit_norm < 0.999999999 or unit_norm > 1.000000001:
        print('Test failed: unit vector not normalized.')
        print('norm =', unit_norm)
        break
    if i == 99:
        print('Success! All tests passed')
'''


for i in range(1):
    det.random_entrance()
    det.gen_capture_locs()
print(det.capture_locs)

'''
start_time = time.monotonic()

det.photon_generation()

end_time = time.monotonic()
duration = end_time - start_time
print('Time to generate all decays:', format_seconds(duration))
'''