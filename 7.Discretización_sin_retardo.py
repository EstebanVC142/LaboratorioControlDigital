from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt

f= 10 #frecuencia de 10 Hz
T = 1/f



G1 = tf(10,[1, 3, 10])
print(G1)

#Por defecto ZOH: Zero-order hold (default)
#Es usado para la planta pero si tengo la FT de un controlador
#uso el metodo bilineal

G1_D = c2d(G1, T, 'zoh')
G2_D = c2d(G1, T, 'foh')
G3_D = c2d(G1, T, 'tustin')


#Estimulamos la ftp para ver la dinamica del sistema
y, t = step(G1)
yd1, tD1 = step(G1_D)
yd2, tD2 = step(G2_D)
yd3, tD3 = step(G3_D)

print(f'zoh: {G1_D}')
print(f'foh: {G2_D}')
print(f'tustin: {G3_D}')

#Modelo para sistemas embebidos

nit = int(10/0.1)
y = np.zeros(nit)
u = np.zeros(nit)
u[10:] = 1 # Apartir de segundo 10 entra el escalon
t = np.arange(0,(nit)*0.1,0.1)

B = G1_D.num[0][0]
A = G1_D.den[0][0]
d = 0

for k in range(4, nit):
    #Escribimos la solución de la ecuacion en diferencias
    y[k] = -A[1]*y[k-1]-A[2]*y[k-2]+B[0]*u[k-1-d] + B[1]*u[k-2-d]

yout, T, xout = lsim(G1_D, u ,t)
plt.figure()
plt.plot(T, yout, t, y, '--')
plt.grid()
plt.title('Z vs ec diferencias')
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()

'''
plt.plot(t, y, label='Continuo')
plt.plot(tD1, yd1, label='zoh')
plt.plot(tD2, yd2, label='foh')
plt.plot(tD3, yd3, label='Tustin')
plt.legend(loc = 'best')
plt.title('Diferentes formas discretizar')
plt.ylabel('Output')
plt.xlabel('Time')
plt.grid()
plt.show()
'''
