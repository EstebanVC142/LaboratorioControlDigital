#Importaciones
import control 
import numpy as np
import matplotlib as plt
from math import pi

#sys : función de transferencia del proceso.
def sampling_band(sys):
    h = control.feedback(sys)
    mag , phase,  w = control.bode(h)
    
    #Mag cuando w = 0
    m0 = mag[0] #Magnitud cuando la frecuencia es w = 0.1 rad/seg
    mwc = 0.707 * m0 #Magnitud de la frecuencia de corte
    
    #Buscar la frecuencia correspondiente a mwc
    index_wc = np.where(mag >= mwc)
    wc = w[index_wc[0][-1]]
    
    wmin = 8 *wc
    wmax = 12* wc
    
    ts_small = (2 * pi)/ wmax
    ts_big   = (2 * pi)/wmin
    
    print('Frecuencia de corte --> ', wc)
    print('Frecuencia de muestreo minima --> ', wmin)
    print('Frecuencia de muestreo maxima --> ', wmax)
    return ((ts_small + ts_big)/2)
#Point 
if __name__ == '__main__':
    num = 8;
    den = [1,10,0]
    sys = control.tf(num, den)
    ts = sampling_band(sys)
    print('ts -->', ts ,'segundos')