

import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
from scipy.integrate import odeint


F = 10 #Frecuencia 10hz
Ts = 1/F

h = tf(10,[1, 3, 10])
#numt, dent = pade(0.25,1)
#theta = tf(numt, dent)
#h1 = series(h, theta)
hd = c2d(h,0.1, 'zoh')

print(hd)


y,t = step(h)


plt.plot(t,y, label='Continuo')
#plt.step(td,yd, label='ZOH')
#plt.step(td1,yd1, label='FOH')
#plt.step(td2,yd2, label='Tustin')
#plt.legend(loc='best')
plt.show()

#  Implementación del modelo en sistemas embebidos

nit = int(10/0.1)
y = np.zeros(nit)
u = np.zeros(nit)
u[10:] =1
t = np.arange(0,(nit)*0.1,0.1)

B = hd.num[0][0]
A = hd.den[0][0]
d = 0
