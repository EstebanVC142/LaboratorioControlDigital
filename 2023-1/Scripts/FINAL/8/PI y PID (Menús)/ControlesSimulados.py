"""

Control P, PI, PID - TCLAB (simulado)
Función de Transferencia Continua:
                1.84 exp(-35.2339s)
    G(s) =  ----------------------
               115.6s + 1
                
Función de Transferencia Discreta:
                 0.2484 z + 0.1222
G(z) = z^(-1) * ---------------------   Ts = 26
                    z^2 - 0.7986 z
@authors: Juan David Corrales Morales
          Esteban Vasquez Cano 
          Daniel Montoya Lopez
"""
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
from Funtions import * 

plt.close()
# Discrete Transfer Function:
Ts   = 26 # Sampling
numz =  np.array([0.2484,  0.1222])   #Numerador
denz =  np.array([1, -0.7986])        #Denominador
d    =  2                   #Retardo
denzd = np.hstack((denz, np.zeros(d)))
Gz   =  tf(numz, denzd, Ts)
print(Gz)

#  Continuous Transfer Function Parameters:
K = 1.84
theta1 = 35.2339
tau = 115.6

# Delay Correction
theta = theta1 + Ts/2

# Reference Temperature °C
Tam = 25             # Initial Temperature
Ref = 40             # Reference Temperature
Tkelvin = 273.15

# Conversion in °K
Tinit = Tam + Tkelvin
Ta=[Tinit]

# Time
tsim = 1000              # Simulation time (sec)
nit = int(tsim/Ts)      # Number of Iterations 

# Vectors
t = np.arange(0, (nit)*Ts, Ts)  # Time Vector
u = np.zeros(nit)           # In Vector (Heater)
y = np.zeros(nit)           # Out Vector  (Temperature)
y[:] = Tinit                # Out = T initial
e = np.zeros(nit)           # Error Vector
q = np.zeros(nit)           # Disturbance  Vector
q[20:] = 0                  

# Setpoint
r = np.zeros(nit)           # Reference Vector
r[:] = Tinit                # Referece = Tinit
r[5:] = Ref + Tkelvin  # Setpoint
Des = f'Escogio el metodo P'
_control = f'P'
# Control     
if metodo == 1: #Ziegler-Nichols
    if control == 1: # P
        kp = tau/(K*theta)
        ti = np.infty
        td = 0
        namecrlt = (' Control Ziegler-Nichols')
        print(f'{Des} {namecrlt}')
    elif control == 2: # PI
        kp = (0.9*tau)/(K*theta)
        ti = 3.3*theta
        td = 0
        namecrlt = ('I Control Ziegler-Nichols')
        print(f'{Des} {namecrlt}')
    else: # PID
        kp = ((1.2*tau)/(K*theta))
        ti = 2*theta
        td = 0.5*theta
        namecrlt =('ID Control Ziegler-Nichols')
        print(f'{Des}{namecrlt}')

elif metodo == 2: # IAE Roveri
    if control == 1: # P
        kp = 1
        ti = np.infty
        td = 0
        namecrlt =(' Control IAE Rovira')
        print(f'{Des}{namecrlt}')
    elif control == 2: # PI
        IAE_PI = (0.984, -0.986, 0.608, -0.707)
        kp = (1/K)*(IAE_PI[0]*(theta/tau)**IAE_PI[1])        
        ti = tau / (IAE_PI[2]*(theta/tau)**IAE_PI[3])
        td = 0
        namecrlt =('I Control IAE Rovira')
        print(f'{Des}{namecrlt}')
    else: # PID
        IAE_PID = (1.435, -0.921, 0.878, -0.749, 0.482, 1.137)
        kp = (1/K)*(IAE_PID[0]*(theta/tau)**IAE_PID[1])        
        ti = tau / (IAE_PID[2]*(theta/tau)**IAE_PID[3])
        td = tau * (IAE_PID[4]*(theta/tau)**IAE_PID[5])
        namecrlt =('ID Control IAE Rovira')
        print(f'{Des}{namecrlt}')

elif metodo == 3: # IAET Roveri
    if control == 1: # P
        kp = 1
        ti = np.infty
        td = 0
        namecrlt =(' Control ITAE Roveri')
        print(f'{Des}{namecrlt}')
    elif control == 2: # PI
        IAET_PI = (0.859, -0.977, 0.674, -0.68)
        kp = (1/K)*(IAET_PI[0]*(theta/tau)**IAET_PI[1])        
        ti = tau / (IAET_PI[2]*(theta/tau)**IAET_PI[3])
        td = 0
        namecrlt =('I Control ITAE Roveri')
        print(f'{Des}{namecrlt}')
    else: # PID
        IAET_PID = (1.357, -0.947, 0.842, -0.738, 0.381, 0.995)
        kp = (1/K)*(IAET_PID[0]*(theta/tau)**IAET_PID[1])        
        ti = tau / (IAET_PID[2]*(theta/tau)**IAET_PID[3])
        td = tau * (IAET_PID[4]*(theta/tau)**IAET_PID[5])
        namecrlt =('ID Control IAET Roveri')
        print(f'{Des}{namecrlt}')

elif metodo == 4: #COHEN-COON
    if control == 1: # P
        kp = (1.03+(0.35*(theta/tau)))*(tau/(K*theta))
        ti = np.infty
        td = 0
        namecrlt =(' Control Cohen-Coon')
        print(f'{Des}{namecrlt}')
    elif control == 2: # PI
        COHEN_COON_PI = (0.9, 0.083, 1.27, 0.6)
        kp = (COHEN_COON_PI[0]+(COHEN_COON_PI[1]*(theta/tau)))*(tau/(K*theta))
        ti = (theta*(COHEN_COON_PI[0]+(COHEN_COON_PI[1]*(theta/tau)))) / (COHEN_COON_PI[2]+(COHEN_COON_PI[3]*(theta/tau)))
        td = 0
        namecrlt =('I Control Cohen-Coon')
        print(f'{Des}{namecrlt}')
    else: # PID
        COHEN_COON_PID = (1.35, 0.25, 0.54, 0.33, 0.5)
        kp = (COHEN_COON_PID[0]+(COHEN_COON_PID[1]*(theta/tau)))*(tau/(K*theta))*0.5
        ti = (theta*(COHEN_COON_PID[0]+(COHEN_COON_PID[1]*(theta/tau)))) / (COHEN_COON_PID[2]+(COHEN_COON_PID[3]*(theta/tau)))
        td = (COHEN_COON_PID[4]*theta) / (COHEN_COON_PID[0]+(COHEN_COON_PID[1]*(theta/tau)))
        namecrlt =('ID Control Cohen-Coon')
        print(f'{Des}{namecrlt}')

else: # PA
    if control == 1: # P
        kp = 1
        ti = np.infty
        td = 0
        namecrlt = (' Control Pole Assignment')
        print(f'{Des}{namecrlt}')
    elif control == 2: # PI
        # Objetivo al 50%
        Tss = (4*tau)*0.6
        M_p = 8
        ep = np.sqrt((np.log(M_p/100)**2) / (np.pi**2 + np.log(M_p/100)**2))
        wn = 3 / (ep*Tss)
        kp = ((2*ep*wn*tau)-1) / K
        ti = (kp*K) / ((wn**2)*tau)
        td = 0
        namecrlt = ('I Control Pole Assignment')
        print(f'{Des}{namecrlt}') 
    else:        
        kp = 1
        ti = np.infty
        td = 0
        namecrlt = ('ID Control Pole Assignment')
        print(f'{Des}{namecrlt}')

    
# Equations Discrete PID Control
q0 = kp*(1+ Ts/(2*ti) + td/Ts)
q1 = -kp*(1- Ts/(2*ti) + td/Ts)
q2 = (kp*td)/Ts

# Closed Control Loop
for k in range(nit):
    
    # Simulation Non-Linear Model 
    if k > 1:
        T1 = cal_Tclab(t[0:k+1], u[0:k+1] - q[0:k+1], Ta[0:k+1], Tinit)
        y[k] = T1[-1]
    
    # Error 
        e[k] = r[k] - y[k]
    
    # Control Law 
        u[k] = u[k-1] + q0*e[k] + q1*e[k-1] + q2*e[k-2]
        
    # Saturation
        if u[k] > 100:
            u[k] = 100
        elif u[k] < 0:
            u[k] = 0

# Conversion in °C
y -= 273.15
r -= 273.15

# Graph 
plt.figure()

plt.subplot(2,1,1)
plt.plot(t,r,'--k',t,y,'r-',linewidth=3)
plt.legend(['Setpoint', 'Output'])
plt.ylabel('Temperature (°C)',fontsize=18)
# plt.xlabel('Time(s)',fontsize=18)
plt.grid()
plt.title(_control + namecrlt,fontsize=22)

plt.subplot(2,1,2)
plt.step(t,u,'b-',linewidth=3)
plt.legend(['Heater'])
plt.ylabel('Power (%)',fontsize=18)
plt.xlabel('Time(s)',fontsize=18)
plt.grid()
save_txt(t, u, y, r, Des + namecrlt)
plt.savefig(Des + namecrlt +".png")
plt.pause(1)
plt.show()