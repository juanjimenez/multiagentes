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
    return np.array([[f1], [f2], [f3]])

def dfun(w):
    '''
    Derivada primera de la función anterior, La defino explícitamente
    '''
    df1 = 10*np.cos(w) + 40*np.cos(2*w)
    df2 = -10*np.sin(w) + 40*np.sin(2*w)
    df3 = -60*np.cos(3*w)
    return np.array([[df1], [df2], [df3]])
    
def ddfun(w):
    '''
    Derivada segunda de la función fun, La defino explícitamente
    '''
    ddf1 = -10*np.sin(w) - 40*np.sin(2*w)
    ddf2 = -10*np.cos(w) + 80*np.cos(2*w)
    ddf3 = 180*np.sin(3*w)
    return np.array([[ddf1], [ddf2], [ddf3]])

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


def din_agen(t,x,N,K,kc,kth,L,v,E,wstr):
    ''' 
    Esta funcion define la dinamica de un agente
    incluye la guia del campo y el efecto de la  cordinación
    Asumo que la dimensión de movimiento en R^3, por tanto (-1)^n = (-1)^3 = -1
    En general, salvo que se diga lo contrario cada columna de valores 
    corresponde a un agente y cada fila a una variable (x,y,z,w),theta)
    '''
    x = x.reshape(5,N)
    c = -L@(x[3,:] - wstr)
    dp = np.zeros((4,N)) #guardar las derivadas de los estados excepto theta
    bXipn = [] #
    Jxip = []  #Jacobiano del campo
    nXipt = []                   
    for i in range(N):
        #calculamos el valor del campo guía en la posición del agente
        xi =x[:,i:i+1]
        w = xi[3,0]
        p = xi[0:3]
        f = fun(w)
        df = dfun(w)
        ddf = ddfun(w)
        Kphi = K@(p-f)
        Kdf  = K@df
        Xi = np.vstack((-df-Kphi,-1+Kphi.T@df + kc*c[i]))
        nXi = np.linalg.norm(Xi)
        Xib = Xi/nXi
        nXip = np.linalg.norm(Xi[0:2])
        nXipt.append(nXip)
        bXipn.append(xi[0:2]/nXip)
        dp1 = np.cos(xi[4]) #x
        dp2 = np.sin(xi[4]) #y
        dp3 = Xi[2]/nXip    #z
        dp4 = Xi[3]/nXip    #w (la coordenada virtual)
        dp[:,i]= (v*np.hstack((dp1,dp2,dp3,dp4))) 
        
        Jpf =np.hstack( \
            (np.vstack((-K,Kdf.T)),\
             np.zeros((4,i)),\
             np.vstack((-ddf + K@df,Kphi.T@ddf-np.ones((1,3))@K@df**2)),np.zeros((4,N-1-i))))
        Jcr = np.hstack((np.zeros((4,3)),np.vstack((np.zeros((3,N)),-L[i,:]))))
        Jxip.append(((np.eye(4)-Xib@Xib.T)@(Jpf+kc*Jcr)/Xib)[0:2,:])
    
    dwbold = dp[3,:]
    dp = np.vstack((dp,np.zeros((1,N))))
    for i in range(N):
        #definimos la dinámica  de theta y la ley de control
        dp[4,i] = -(bXipn[i].T@E@Jxip[i]@np.append(dp[0:3,i:i+1],dwbold)/nXipt[i]\
            -kth*dp[0:2,i]@E@bXipn[i])[0]#dot theta       
    return dp.reshape(5*N)
    
###########################Integracion del modelo#############################


#Parámetro de simulación 
#dimensiones del espacio 3D
N = 4         #nº de agentes 
K =np.diag([1,1,2]) #ganancias
kc = 200
kth = 1
L = grafo_L(N) #laplaciana daisy chain de la formacion
v = 2 #velocidad fija de los uniciclos
#coordenadas iniciales de los robots 
#pos = np.random.rand(N, n)*100 #filas: dimensiones
E = np.array([[0,-1],[1,0]])                          #columnas: nº de robots
p = np.random.randint(-50, 51, size=(N, 3))
theta = 2*np.pi*np.random.random((N,1))

#añadimos a la matriz de posiciones la coordenada virtual w 
w = np.ones((N,1)) #ejemplo: todos valen 1
wstr = np.arange(0,2*np.pi,2*np.pi/N)    #distribución deseada
x = np.hstack((p,w,theta)).reshape(5*N)
t = (0,2)
sol = sl(din_agen,t,x,method='LSODA',\
         args=(N,K,kc,kth,L,v,E,wstr),max_step=0.2)

#representación gráfica 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

lista = np.arange(0,(4)*N+1,(4))
for i in lista[:-1]: 
    ax.plot(sol.y[i,:], sol.y[i+1,:], sol.y[i+2,:])
    ax.scatter(sol.y[i,0],sol.y[i+1,0], sol.y[i+2,0], marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Trayectoria de los agenetes')
plt.show()
