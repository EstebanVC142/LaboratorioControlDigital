'''
Spyder Editor

Equipo: ¤ Juan David Corrales
        ¤ Daniel Montoya Lopez
        ¤ Esteban Vásquez Cano       
'''

import control 
import numpy as np
from math import pi

#%% Función para encontrar el periodo de muestreo por el metodo de ancho de banda

#sys : función de transferencia del proceso.

def MuestreoAnchoDeBanda(sys):
    h = control.feedback(sys)
    mag, phase,  w = control.bode(h)
    
    #Mag cuando w = 0
    m0 = mag[0] #Magnitud cuando la frecuencia es w = 0.1 rad/seg
    mwc = 0.707 * m0 #Magnitud de la frecuencia de corte
    
    #Buscar la frecuencia correspondiente a mwc
    index_wc = np.where(mag >= mwc)
    wc = w[index_wc[0][-1]]
    
    wmin = 8 *wc
    wmax = 12* wc
    
    ts_small = (2 * pi)/wmax
    ts_big   = (2 * pi)/wmin
    
    print('Frecuencia de corte --> ', wc)
    print('Frecuencia de muestreo minima --> ', wmin)
    print('Frecuencia de muestreo maxima --> ', wmax)
    return ((ts_small + ts_big)/2)

#%% Funcion para encontrar el tiempo de establecimiento

def MuestreoSN(tss):
    minima = tss/20
    maxima = tss/10
    
    return (minima+maxima)/2

#%% Encontrado en periodo de muestreo para G(s)1 por ambos metodos

Gs1 = control.tf([1, -0.5],[1, 0.2])

Gs1Poles = control.poles(Gs1)
tss1 = (-1/Gs1Poles.real[0])*4
SN1 = MuestreoSN(tss1)

print('='*20, 'Para G(s)1 tenemos','='*20, '\n')
print(Gs1)
print('-'*10,'Por ancho de banda', '-'*10,'\n')

if __name__ == '__main__':
    ts1 = MuestreoAnchoDeBanda(Gs1)
    print('El tiempo de muestreo para G(s)1 es: -->', ts1 ,'segundos')
print('\n')
print('-'*10,'Por metodo empirico de Ziegler y Nichols', '-'*10, '\n')
print('El periodo de muestero por metodo de ZN es: ', SN1, ' segundos')   
print('\n')
#%% Encontrado en periodo de muestreo para G(s)2 por ambos metodos

Gs2 = control.tf([9],[1, 6, 9])

Gs2Poles = control.poles(Gs2)
tss2 = (-1/Gs2Poles.real[0] - 1/Gs2Poles.real[1])*4
SN2 = MuestreoSN(tss2)

print('='*20, 'Para G(s)2 tenemos','='*20, '\n')
print(Gs2)
print('-'*10,'Por ancho de banda', '-'*10, '\n')

if __name__ == '__main__':
    ts2 = MuestreoAnchoDeBanda(Gs2)
    print('El tiempo de muestreo para G(s)2 es: -->', ts2 ,'segundos')
print('\n')
print('-'*10,'Por metodo empirico de Ziegler y Nichols', '-'*10, '\n')
print('El periodo de muestero por metodo de ZN es: ', SN2, ' segundos')
print('\n')  
#%% Encontrado en periodo de muestreo para G(s)3 por ambos metodos

Ga = control.tf([2.6],[1, 2])
Gb = control.tf([1],[1, 4])
Gc = control.tf([1],[1, 4])

Gs3 = Ga * Gb * Gc

Gs3Poles = control.poles(Gs3)
tss3 = (-1/Gs3Poles.real[0] - 1/Gs3Poles.real[1] - 1/Gs3Poles.real[2])*4
SN3 = MuestreoSN(tss3)

print('='*20, 'Para G(s)3 tenemos','='*20, '\n')
print(Gs3)
print('-'*10,'Por ancho de banda', '-'*10, '\n')

if __name__ == '__main__':
    ts3 = MuestreoAnchoDeBanda(Gs3)
    print('El tiempo de muestreo para G(s)3 es: -->', ts3 ,'segundos')
print('\n')
print('-'*10,'Por metodo empirico de Ziegler y Nichols', '-'*10, '\n')
print('El periodo de muestero por metodo de ZN es: ', SN3, ' segundos')


