import numpy as np
from scipy.integrate import solve_ivp
import time

menu1 = """ 
Method of Controller Adjustment:
    [1] Ziegler - Nichols
    [2] IAE
    [3] IAET
    [4] Cohen - Coon
    [5] Pole Assignment
Select 1-5:  """
menu2 = """
Controller Type:
    [1] P
    [2] PI
    [3] PID
Select 1-3: """

# Method
metodo = int(input(menu1))
while not (0 < metodo < 6):
    print("\n Error")
    metodo = int(input(menu1))

# Control   
control = int(input(menu2))
while not (0 < control < 4):
    print("\n Error")
    control = int(input(menu2))

# Save Data
def save_txt(t, u1, y1, r1, nombreArchivo = "experimento"):
    data = np.vstack((t,u1,y1,r1))  # vertical stack
    data = data.T                 # transpose data
    top = 'Tiempo (sec), Calentador 1 (%), ' \
        + 'Temperatura 1 (degC),' + 'Referencia '
    np.savetxt(nombreArchivo + '.txt', data,delimiter=',', header=top, comments='')

# EDO Tclab
def TC_LAB (t,x,Qi,tinit):
       
    
    cp = 500            # Heat Capacity
    m = 0.004           # Mass
    sigma = 5.6e-8      # Stefan Boltzmann Constant
    A = 1.2e-3          # Area
    ε = 0.9             # Emissivity
    alpha = 0.01388679  # Heater Factor
    U = 5.80969803      # Heat Transfer Coefficient
    Ta = tinit          # Initial Temperature
    
    T = x[0]            #Renombrando los estados de EDO 

    #EDO
    dx1dt= (alpha*Qi+ U*A* (Ta-T)+ ε*sigma*A* ((Ta**4)-(T**4))) / (m*cp)
    return dx1dt

# Calculate Temperature
def cal_Tclab(t, u, x0,tinit):
    y0 = list(x0)
    ym = np.ones(len(t))*x0[0]
    for i in range(len(t)-1):
        tspan = [t[i], t[i+1]]  
        sol= solve_ivp(TC_LAB, tspan, y0, method = 'RK45', args = (u[i],tinit)) 
        y0 = sol.y[:, -1]
        ym[i+1]= y0[0]
    return ym

# Funtion Delay
def delay_time(sleep_max, prev_time):
    sleep = sleep_max - (time.time() - prev_time)
    if sleep >= 0.01:
        time.sleep(sleep - 0.01)
    else:
        time.sleep(0.01)
        
    # Record time and change in time
    t = time.time()
    return t
def save_txt(t, u1, y1, r1, nombreArchivo = "experimento"):
    data = np.vstack((t,u1,y1,r1))  # vertical stack
    data = data.T                 # transpose data
    top = 'Tiempo (sec), Calentador 1 (%), ' \
        + 'Temperatura 1 (degC),' + 'Referencia '
    np.savetxt(nombreArchivo + '.txt', data,delimiter=',', header=top, comments='')