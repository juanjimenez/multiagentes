# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:48:21 2024
Ejmplillo de sistema multiagente que converge a una treayectoria.
La dinamica de los agentes corresponde a u uniciclo
@author: abierto
"""

import numpy as np
from scipy.integrate import solve_ivp as sl
import matplotlib.pyplot as plt


###################funciones para definir la trayectoria######################
def fun(w):
    '''
    Esta función debe definir una curva en 3D que seguirán los agentes
    debe estar definida en funcion de la coordenada w (augmented field)
    '''
    #f1 = 15*np.sin(2*w)
    #f2 = 30*np.sin(w)*(np.sqrt(0.5*(1-0.5*(np.sin(w))**2)))
    #f3 = 3+5*np.cos(2*w)
    f1 = 10*np.sin(w) + 20*np.sin(2*w)
    f2 = 10*np.cos(w)-20*np.cos(2*w)
    f3 = -20*np.sin(3*w)
    return np.array([f1, f2, f3])

def dfun(w):
    '''
    Derivada primera de la función anterior, La defino explícitamente
    '''
    df1 = 10*np.cos(w) + 40*np.cos(2*w)
    df2 = -10*np.sin(w) + 40*np.sin(2*w)
    df3 = -60*np.cos(3*w)
    return np.array([df1, df2, df3])
    
def ddfun(w):
    '''
    Derivada segunda de la función fun, La defino explícitamente
    '''
    ddf1 = -10*np.sin(w) - 40*np.sin(2*w)
    ddf2 = -10*np.cos(w) + 80*np.cos(2*w)
    ddf3 = 180*np.sin(3*w)
    return np.array([ddf1, ddf2, ddf3])

#matriz lalpaciana de un grafo conexion en daisy chain
def grafo_L(N):
    '''
    Cada agente solo se relaciona con el anterior y el siguiente, formando 
    una cadena cerrada

    '''
    L =2*np.eye(N)
    sd = -np.ones([N-1])
    L = L + np.diag(sd,1)+np.diag(sd,-1)
    
    # Llenar las esquinas con -1
    L[0, -1] = -1
    L[-1, 0] = -1
    return L

def din_agen(i,xi,K,kc,C,v):
    ''' 
    Esta funcion define la dinamica de un agente
    incluye la guia del campo y el efecto de la  cordinación
    Asumo que la dimensión de movimiento en R^3, por tanto (-1)^n = (-1)^3 = -1
    '''
    #calculamos el valor del campo guía en la posición del agente
    w = xi[3]
    p = xi[0:3]
    f = fun(w)
    df = dfun(w)
    ddf = ddfun(w)
    Kphi = K@(p-f)
    Kdf  = K@df
    
    xi = -df-Kphi
    xi = np.append(xi,-1+Kphi@df + kc*C)
    nxip = np.linalg.norm(xi[0:2])
    bxip = xi[0:2]/nxip
    Jxip = np.append(-K,Kdf,axis=0)
    #definimos la dinámica y la ley de control
    dp1 = v*np.cos(p[5])
    dp2 = v*np.sin(p[5])
    dp3 = v*xi[2]/nxip
    dp4 = v*xi[4]/nxip
    
    
    
    