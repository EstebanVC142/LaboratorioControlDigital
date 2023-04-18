# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:28:18 2023

@author: Esteban VC
"""

import tclab_cae.tclab_cae as tclab
import time 

lab = tclab.TCLab_CAE()

print('Turn ON the heaters for 20 seconds')
lab.Q1(100) # Percentage 0 - 100%
lab.LED(100) # Percentage 0 - 100%

for i in range(40):
    print('Time:', i, 'Temperature 1:', lab.T1, 'Temperature atmosphere: ', lab.T3, 'Current:', lab.I1)   
    time.sleep(1)
    
lab.Q1(0) # Percentage 0 - 100%
lab.LED(0) # Percentage 0 - 100%
lab.close()
