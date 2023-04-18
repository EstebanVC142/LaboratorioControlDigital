# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:57:53 2022

@author: Sergio
"""
import numpy as np
import matplotlib.pyplot as plt


def create_prbs(ValUinit, ValAmpli, ValDecal, ValLgReg, ValDivi, Nsamp, Tappli):
    """
    
    CREATE_PRBS  is used for the generation of a PRBS signal

       prbs = create_prbs(ValUinit, ValAmpli, ValDecal, ValLgReg, ValDivi, Nsamp, Tappli)             

   "Entry parameters" are :
	ValUinit  : Initial steady state
    ValAmpli  : Magnitude
    ValDecal  : Add-on DC component
    ValLgReg  : Register length
    ValDivi   : Frequency divider
    Nsamp     : Number of samples
    Tappli    : Application instant 
	  
 
 

	              ____  Valdecal + ValAmpli         __________      ____
                 |    |                            |          |    |
 Valdecal       -|----|--------                    |          |    |
                 |    |____________________________|          |____|
                 |
                 |
 ini ____________|
                                                   |--------->|
     |-Tappli -->|                        ValReg * ValDivi 
     

     |---------- Nsamp ------------------------------------------------->|
                        
    
	"Exit parameter" is  :
    prbs : prbs sequence created by PRBS algo

    """
    k1 = ValLgReg - 1
    k2 = ValLgReg 
    
    if ValLgReg == 5:
        k1 = 3
    elif ValLgReg == 7: 
        k1 = 4
    elif ValLgReg == 9:   
        k1 = 5
    elif ValLgReg == 10: 
        k1 = 7
    elif ValLgReg == 11: 
        k1 = 9    

    sbpa = [1]*11
    
    prbs = [0] * (Nsamp + Tappli)
    
    for i in range(Tappli):
       prbs[i] = ValUinit;
    # PRBS sequence generation 
    i=Tappli+1;
    while i <= Nsamp:
        uiu = -sbpa[k1]*sbpa[k2]
        if ValLgReg == 7:
            uiu = -sbpa[1]*sbpa[2]*sbpa[4]*sbpa[7]
        
        j=0
        while j <= ValDivi:
            prbs[i] = uiu * ValAmpli + ValDecal
            i += 1
            j += 1
        
        for j in range(ValLgReg,0 , -1):
            sbpa[j] = sbpa[j-1]
        
        sbpa[0] = uiu;
        
    return prbs


    
if __name__ == '__main__':
    tPRBS = 2000; 
    prbs = create_prbs(0, 5, 20, 10, 50, tPRBS, 30);
    t = np.linspace(0, len(prbs),len(prbs))
    plt.plot(t, prbs, linewidth = 3)    
