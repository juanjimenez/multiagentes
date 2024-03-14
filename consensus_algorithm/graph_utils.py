#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:16:10 2021
%some resources for dealing with graphs
@author: juanjimenez
"""
import numpy as np

def incidence(V,E):
    """ Return the incidence matrix of a graph
    imputs:
        V, list of vertices (1,2,3...N)
        E, List of edges, each edge is a tuple (e_head, e_tail)
    """
     
    B = np.zeros([len(V),len(E)])
    
    for i in V:
     for j in enumerate(E):
         if i==j[1][0]:
            B[i,j[0]] = 1
         elif i==j[1][1]:
            B[i,j[0]] = -1
            
         
    return B

def Laplacian(w,B,E=[]):
    """Return the Laplacian matrix asociated to a graph
         There are different options:
             1: E=[] (default)
             in this case B is interpreted as an incidence matrix
              1.1. w is a full matrix (e) e:=number of edges
                  wij weight of each edge (directed graph ¿?) 
              1.2. w is a vector wk is the weight of edge k (indirected graph)
              1.3. w is a number; all edges share the same weight
             2: E = list of edges
             in this case B should be a range(0,N) N = number of vertices, i,e
             is a list of vertices, 
             The incidence matrix is first calculated and then, the laplacian 
             with the same cases as in 1
    """
    
    if len(E) != 0:
        B = incidence(B,E)
        
    if type(w)!=np.ndarray: #it's a number
        Dw = w*np.eye(B.shape[1])
    elif len(w)== B.shape[1]:
        Dw = np.diag(w)
    else:
        Dw = w
    return(np.dot(np.dot(B,Dw),B.T)) #Laplacian returned
    
        
    
    
            
            
        
        
    