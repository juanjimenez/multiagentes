#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:23:01 2021

@author: juanjimenez
playing with consesus algorithm
"""
import numpy as np
import graph_utils
from matplotlib import pyplot as pl
from scipy import linalg as lng

N = 6 #number of agents

#graph building 
V = range(N) #vertices
#Undirected graph no cicles
E = [(0,1),(0,3),(1,2),(1,4),(4,5)]

w = 1. #weight for the edges (equal for all edges)
B = graph_utils.incidence(V,E)
L = graph_utils.Laplacian(w,B)

print('incidence\n',B,'\n','laplacian\n',L)


#a configuration of interest
pcm = np.array([[1.],[1.]]) #centroid
uno = np.ones([N,1]) #the staked '1'
# Desires Position of agents 
pstar0 = np.array([[0.],[1.]])
pstar1 = np.array([[-1.],[0.]])
pstar2 = np.array([[-1.],[-1.]])
pstar3 = np.array([[0.],[-1.]])
pstar4 = np.array([[1.],[0.]])
pstar5 = np.array([[1.],[1.]])
pstart =np.kron(uno,pcm)+\
np.concatenate((pstar0,pstar1,pstar2,pstar3,pstar4,pstar5),axis=0)
for i in range(0,2*N,2):
    pl.plot(pstart[i],pstart[i+1],'x')

#Stacked Incidence matrix
B_bar = np.kron(B,np.eye(2))
#Stacked Laplacian
L_bar = np.kron(L,np.eye(2))


#initial random positions
p = np.random.rand(12,1)*3
for i in range(0,2*N,2):
    pl.plot(p[i],p[i+1],'.')

#Consensus algorithm    
step =0.01
t = 0
while t < 10:
    p = p -np.dot(L_bar,(p-pstart))*step
    for i in range(0,2*N,2):
        pl.plot(p[i],p[i+1],'.',markersize=1)
    t = t + step

#for i in range(0,2*N,2):
#    pl.plot(p[i],p[i+1],'.')   

pl.plot(p[0::2],p[1::2])

# zstar
zstar = np.dot(B_bar.T,pstart)

#vstart
vstar = np.array([[1],[2]])
kappa = np.linalg.norm(vstar)
vstar_n = vstar/kappa


#lng.solve()
#