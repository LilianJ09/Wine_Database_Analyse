# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:53:47 2024

@author: basti

On a 3 sites industriels : A, B et C.
Sur chaque site, on a mesuré la concentration d'un certain polluant [ppm]

Question :
    La concentration moyenne en polluant diffère-t-elle entre les sites ?
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols

# dico = {
#         'A' : [23, 34, 65, 76, 85, 34, 23, 56, 76, 45],
#         'B' : [56, 45, 34, 65, 34, 76, 24, 65, 34, 23],
#         'C' : [34, 76, 43, 23, 37, 37, 73, 85, 96, 46]
#         }
# Data = pd.DataFrame(dico)

# A = Data['A']
# B = Data['B']
# C = Data['C']

# cov_matrix = Data.cov()
# print(cov_matrix)

dico = {
        'Sites' : ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
        'Concentrations' : [23, 34, 65, 76, 85, 34, 23, 56, 76, 45, 56, 45, 34, 65, 34, 76, 24, 65, 34, 23, 34, 76, 43, 23, 37, 37, 73, 85, 96, 46]
        }
Data = pd.DataFrame(dico)

sites = Data['Sites']
concentrations = Data['Concentrations']

# Extraire les concetrations du site A, B et C
A = Data[Data['Sites'] == 'A']['Concentrations']
B = Data[Data['Sites'] == 'B']['Concentrations']
C = Data[Data['Sites'] == 'C']['Concentrations']
# On récupère les moyennes puis les écarts-types
A_mean = A.mean()
B_mean = B.mean()
C_mean = C.mean()

A_std = A.std()
B_std = B.std()
C_std = C.std()
# liste de toutes les moyennes et moyenne globale
all_mean = np.array([A_mean, B_mean, C_mean])
global_mean = all_mean.mean()

# somme des carrés intra-classes (varriance multiplié par le nombre d'élément moins 1)
SSA = np.sum((A - A_mean)**2)
SSB = np.sum((B - B_mean)**2)
SSC = np.sum((C - C_mean)**2)
SSintra = SSA + SSB + SSC
# somme des carrés extra-classes
SSextra = np.sum(10*(global_mean - all_mean)**2)
# somme des carrés totale
SSt = SSintra + SSextra

# nombre de classe
J = 3

ddl_J1 = J - 1
ddl_nJ = 30 - J

# Carrés moyens (SS/ddl)
F_num = SSextra/ddl_J1
F_den = SSintra/ddl_nJ

F = F_num/F_den

# test Anova
f_value, p_value = stats.f_oneway(A, B, C)

# Interprétation de l'ANOVA
alpha = 0.05
if p_value < alpha:
    print("La concentration moyenne en polluant diffère significativement entre les sites.")
else:
    print("Aucune différence significative de la concentration moyenne en polluant entre les sites.")

#-------- TABLE ANOVA --------#
# Définir le modèle
model = ols('Concentrations ~ Sites', data=Data).fit()

# Effectuer l'ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

