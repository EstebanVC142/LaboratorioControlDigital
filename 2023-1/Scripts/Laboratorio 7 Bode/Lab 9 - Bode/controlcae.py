# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:34:19 2022

@author: Sergio
"""

import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
from control import *

def margin_plot(sys):
    """margin_plot(sysdata)

    Calculate gain and phase margins and associated crossover frequencies
    and plot the gain margin and phase margin marked with solid vertical lines. 
    The dashed vertical lines indicate the locations of Wcg, the frequency where the gain margin is measured, 
    and Wcp, the frequency where the phase margin is measured.

    Parameters
    ----------
    sysdata : LTI system or (mag, phase, omega) sequence
        sys : StateSpace or TransferFunction
            Linear SISO system representing the loop transfer function
        mag, phase, omega : sequence of array_like
            Input magnitude, phase (in deg.), and frequencies (rad/sec) from
            bode frequency response data

    Returns
    -------
    gm : float
        Gain margin
    pm : float
        Phase margin (in degrees)
    wcg : float or array_like
        Crossover frequency associated with gain margin (phase crossover
        frequency), where phase crosses below -180 degrees.
    wcp : float or array_like
        Crossover frequency associated with phase margin (gain crossover
        frequency), where gain crosses below 1.

    Margins are calculated for a SISO open-loop system.

    If there is more than one gain crossover, the one at the smallest margin
    (deviation from gain = 1), in absolute sense, is returned. Likewise the
    smallest phase margin (in absolute sense) is returned.

    Examples
    --------
    >>> sys = tf(1, [1, 2, 1, 0])
    >>> gm, pm, wcg, wcp = margin_plot(sys)
    
    by: Sergio Andres Castaño Giraldo
    """
    mag, phase, omega = bode(sys)
    gm, pm, Wcg, Wcp = margin(sys)
    plt.subplot(211)
    plt.scatter(Wcg, -20*np.log10(gm), color ='r')
    plt.plot((Wcg, Wcg), (-20*np.log10(gm),0), '-r', linewidth=3 )
    plt.plot((Wcg, Wcg), (-20*np.log10(gm),min(20*np.log10(mag))), ':r' )
    plt.plot((omega[0],omega[-1]),(0,0),':r')
    plt.plot((Wcp, Wcp), (min(20*np.log10(mag)),0), ':r' )

    plt.subplot(212)
    plt.plot((Wcg, Wcg), (-180,20*np.log10(mag[0])), ':r' )
    plt.plot((omega[0],omega[-1]),(-180,-180),':r')
    plt.scatter(Wcp, -180+pm, color ='r')
    plt.plot((Wcp, Wcp), (-180+pm,-180), '-r',linewidth=3 )
    plt.plot((Wcp, Wcp), (-180+pm,max(phase)), ':r' )
    return gm, pm, Wcg, Wcp