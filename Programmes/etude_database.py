import pandas as pd #dataframe with mixed types
import numpy as np #array with same type elements
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
import scipy.stats as stats

from statsmodels.formula.api import ols
from matplotlib.colors import LinearSegmentedColormap

import our_function as my

#variables continues (age) boxplot, histogrammes, qqplot
#variables discretes (enfants) diagrammes en batons
#variables catégorielles (urbaine) camembert, diagramme à bandole

# creation d'une variable categorielle à partir d'une variable de float
# Définition d'une fonction pour mapper les valeurs de la colonne à "acide" ou "basic" en fonction de la condition

#load data form csv file
num_data_red = my.extract_red_Wine()

#load data form csv file
num_data_white = my.extract_white_Wine()

print(num_data_red)

# Appliquer la fonction à la colonne "fixed acidity"
all_data_red = num_data_red.copy()
all_data_red['categorized acidity'] = all_data_red['fixed acidity'].apply(my.map_acidity)
all_data_red['categorized quality'] = all_data_red['quality'].apply(my.map_quality)

all_data_white = num_data_white.copy()
all_data_white['categorized acidity'] = all_data_white['fixed acidity'].apply(my.map_acidity)
all_data_white['categorized quality'] = all_data_white['quality'].apply(my.map_quality)
# Affichage du nombre d'occurrences de chaque catégorie
#print(num_data_red['fixed acidity'].value_counts())

all_data_red.info()
print(all_data_red.describe())#to get all the statistic informations
print(all_data_red.isnull().sum())#count the null value of all dataset fields

num_data_white.info()
print(num_data_white.describe())
print(num_data_white.groupby('quality').mean())#to look at all field inluences on quality value and find average values for each parameters

'''
#give tuples from dictionnaries whose contains keys and value
for i in num_data_red.items():
    my.plot_boxplot_histogram_qqplot(i[1], i[0], i[0], 'Count')
'''
'''
my.plot_boxplot_histogram_qqplot(pH, 'pH', 'pH', 'Count')
#volatile acidity , ph and density look as normal distribued - non normales a causes des valeurs particulières
my.shapiro_wilk_test(volatile_acidity, 'volatile_acidity')
my.shapiro_wilk_test(pH, 'pH')
my.shapiro_wilk_test(density, 'density')
'
my.bar_plot_discrete_variables(num_data_white['quality'], 'White wine : quality', 'quality')
my.bar_plot_discrete_variables(num_data_red['quality'], 'Red wine : quality', 'quality')

# matrice 12 par 12
M_corr_red = num_data_red.corr()
#my.heatMap(M_corr_red, 'Vin rouge')#print correlation matrice

M_corr_white = num_data_white.corr()
#my.heatMap(M_corr_white, 'Vin blanc')

#-------- TABLE DE L'ANOVA --------#

#on sépare la population en 3 groupe en fonction de la qualité :
#H0 : les densités moyennes des 3 groupes sont égales
#H1 : Au moins 2 groupes on des densités moyennes différentes
model = ols('alcohol ~ categorized_quality', data=all_data_red).fit()# Définir le modèle
anova_table = sm.stats.anova_lm(model, typ=2)# Effectuer l'ANOVA
print(anova_table)
A = all_data_red[all_data_red['categorized quality'] == 'good']['density']
B = all_data_red[all_data_red['categorized quality'] == 'medium']['density']
C = all_data_red[all_data_red['categorized quality'] == 'bad']['density']
f_value, p_value = stats.f_oneway(A, B, C)
'''
model = ols("all_data_red['alcohol'] ~ all_data_red['categorized quality']", data=all_data_red).fit()# Définir le modèle
anova_table = sm.stats.anova_lm(model, typ=2)# Effectuer l'ANOVA
print(anova_table)
quality_alcohol = {
    'good' : all_data_red[all_data_red['categorized quality'] == 'good']['alcohol'],
    'medium' : all_data_red[all_data_red['categorized quality'] == 'medium']['alcohol'],
    'bad' : all_data_red[all_data_red['categorized quality'] == 'bad']['alcohol']
}
f_value, p_value = stats.f_oneway(quality_alcohol['good'], quality_alcohol['medium'], quality_alcohol['bad'])
'''
print(f'f_value : {f_value:0.4}, p_value : {p_value:0.4}')
# Interprétation de l'ANOVA
alpha = 0.05
if p_value < alpha:
    print("Au risque de 5%, la concentration moyenne en alchool diffère significativement entre les vins des 3 indexs de qualité.\n")
else:

    print("Au risque de 5%, il n'y a aucune différence significative des concentrations moyennes en alchool entre les vins des 3 indexs de qualité.\n")

'''
#save plot in a folder
# All box plot of all variables
for i in num_data_red.items():
    my.plot_boxplot_histogram_qqplot(i[1], 'Red wine : '+ i[0], i[0], 'Count')
    
for i in num_data_white.items():
    my.plot_boxplot_histogram_qqplot(i[1], 'White wine : '+ i[0], i[0], 'Count')
'''

#regression lineaire simple pour predire la densité (Y) en fonction de la fixed acidity
X = all_data_red['fixed acidity']
Y = all_data_red['density']
my.linear_simple_regression(X, Y, 'fixed acidity', 'density')