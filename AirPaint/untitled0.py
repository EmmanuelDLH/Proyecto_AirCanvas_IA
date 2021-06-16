# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:22:26 2021

@author: Emmanuel_Ledezma_H
"""

#[[4.2785986e-27 7.6619977e-01 2.3379944e-01 7.2321762e-07 0.0000000e+00]]
#False True True False True
lista = [4.2785986e-27, 7.6619977e-01, 2.3379944e-01, 7.2321762e-07, 0.0000000e+00]

p = str(lista[1])
porcentajeValido1 = str(lista[1]).find("e") == -1 #or str(lista[1]).find("+") != -1

print(porcentajeValido1)