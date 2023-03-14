# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:32:09 2022

@author: Sergio
"""
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
#from controlcae import *



plt.close()
# Función de Transferencia Discreta (Profesor)
Ts   =  30                   #Periodo de Muestreo
numz =  np.array([0.06448, 0.053144])   #Numerador
denz =  np.array([1, -0.8869,0])        #Denominador
d    =  1                   #Retardo
denzd = np.hstack((denz, np.zeros(d)))
Gz   =  tf(numz, denzd, Ts)
print(Gz)


#cG --> controlador (k) y planta
#cG = series(k, G)
#bode(cG)
# Analisis de Estabilidad
gm, pm, Wcg, Wcp = margin(Gz)
# Margin --> encuentra los margenes de ganancia y fase
# cruces de fase y ganancias
#gm, pm, Wcg, Wcp = margin_plot(Gz)
print('Margen de Ganancia: ', gm)
print('Margen de Fase: ', pm)
print('Frecuencia de Cruce de Fase: ', Wcg)
print('Frecuencia de Cruce de Ganancia: ', Wcp)

#close loop
H = control.feedback(Gz, 1)
print(H)

y, t = step(H)
plt.figure()
plt.plot(t, y)
plt.grid()