#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from EulerFuncional import *
from TrapecioFuncional import *


# In[ ]:


def f(t,y):
    #return (t*(np.e**(3*t)))-(2*y)   # TP1 b)
    #return 1 + ((t-y)**2)           # TP1 a)
    #return (2-(2*t*y))/((t**2)+1)   # Ejemplo Euler
    return 2*y/t+t**2*np.e**t

def yReal(t):
    #return (1/5)*t*(np.e**(3*t))-(1/25)*(np.e**(3*t))+((1/25)*(np.e**(-2*t)))    # TP1 b)
    #return t + (1/(1-t))                                                        # TP1 a)
    #return ((2*t)+1)/((t**2)+1)                                                 # Ejemplo Euler
    return (1/2)*np.sin(2*t)-(1/3)*np.cos(3*t)+(4/3)


# In[ ]:


def Heun(faprox,freal,h,I,y0,grafico,Epsilon,maxiter):
    '''
    Método Predictor - Corrector para aproximar funciones
    Como método predictor utiliza el Método de Euler Explícito
    Como método corrector utiliza el Método de los Trapecios
    
    faprox = Funcion a Aproximar
    freal = Funcion Real
    h = Tamaño del paso
    I = Par [a,b] para calcular los pasos
    y0 = Valor iniciar conocido
    grafico = Decido Realizar el o no el grafico
    '''
    
    p0 = Euler(f,yReal,h,[I[0],I[0]+h],y0,False).loc[1]
    print(p0)
    numiter = 1
    yaprox1 = 9999.9
    yaprox0 = float(p0['yAprox'])
    while numiter<maxiter:        
        t0 = float(p0['t'])
        y0 = float(p0['yAprox'])
        ynew = trapecio(f,yReal,h,[t0,t0+h],y0,False).loc[1]
        yaprox1 = float(ynew['yAprox'])
        if abs(yaprox1-yaprox0)<Epsilon:
               break
        yaprox0 = yaprox1
        p0 = ynew
        p0['t'] = t0
        print("PRINTING p0\n",p0)
        numiter += 1
    print("PRINTING y0 FINAL\n",p0)
               


# In[ ]:


I = [1,2]
h = 0.10
y0 = 0
epsilon = 0.01
Heun(f,yReal,h,I,y0,True,epsilon,50)

