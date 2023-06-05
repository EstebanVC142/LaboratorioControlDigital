from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Programamos la EDO
# x: entrada de ecuacion diferencial
# t: Tiempo de integracion
# u: Parametro de entrada (Q de nuestro transistor)
def EDO_TCLAB(t, x, Q, Tinit):
    
    U = 5.80969803 # starting at 10.0 W/m^2-K
    alpha = 0.01388679 # starting as 0.01 W / % heater
    Tamb = Tinit   #Kelvin
    A = 0.0012      #Transistor area
    m = 0.004       #Transistor mass
    Cp = 500        #heat capacity
    ε = 0.9         #Emissivity
    sigma = 5.67e-8        #Stefan-boltzman constant     
    
    #Renombrando los estados de EDO
    T = x[0]
        
    #Nonlinear Energy Balance
    dTdt = ((U*A*(Tamb-T)/(m*Cp)) \
            + ε * sigma * A * (Tamb**4 - T**4) \
            + alpha*Q)

    return dTdt

#calculo del modelo
def temperature_tclab(t, Q, Ta, Tinit):
    y0 = list(Ta)
    ym = np.ones(len(t))*Ta[0]
    for i in range(len(t)-1):
        tspan = [t[i], t[i+1]]   # Diferencial de Tiempo
        sol = solve_ivp(EDO_TCLAB, tspan, y0, method = 'RK45', args = (Q[i],Tinit)) 
        y0 = sol.y[:, -1]
        ym[i+1]= y0[0]
    return ym 

def save_txt(t, u1, y1, r1, nombreArchivo = "experimento"):
    data = np.vstack((t,u1,y1,r1))  # vertical stack
    data = data.T                 # transpose data
    top = 'Tiempo (sec), Calentador 1 (%), ' \
        + 'Temperatura 1 (degC),' + 'Referencia '
    np.savetxt(nombreArchivo + '.txt', data,delimiter=',', header=top, comments='')