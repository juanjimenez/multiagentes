#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:03:51 2023
Pruebas con picos
@author: juanjimenez
"""
import picos
import numpy as np

P = picos.Problem()
A = np.array([[0,1],[-2,-1]])
B = np.array([[0],[1]])
p = picos.SymmetricVariable("p",2)

A = picos.Constant('A', A)
P.add_constraint(-p<<0)
s = (A.T*p+p*A)
P.add_constraint(-s>>0)
