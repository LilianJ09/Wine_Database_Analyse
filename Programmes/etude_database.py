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
def map_acidity(value):
    if value > 7:
        return "acide"
    else:
        return "basic"
    
def map_quality(value):
    if value >= 7:
        return "good"
    elif value >= 5:
        return "medium"
    else:
        return "bad"

#load data form csv file
num_data_red = pd.read_csv("../Bases de données/winequality-red.csv", sep=';', decimal='.')

#load data form csv file
num_data_white = pd.read_csv("../Bases de données/winequality-white.csv", sep=';', decimal='.')

#define data columns as variables for easy access
columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide", "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

#extraction de colonnes par identifiant
fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality = (num_data_red[col] for col in columns)

print(num_data_red)

# Appliquer la fonction à la colonne "fixed acidity"
all_data_red = num_data_red.copy()
all_data_red['categorized acidity'] = all_data_red['fixed acidity'].apply(map_acidity)
categorized_acidity = all_data_red['categorized acidity']
all_data_red['categorized quality'] = all_data_red['quality'].apply(map_quality)
categorized_quality = all_data_red['categorized quality']
# Affichage du nombre d'occurrences de chaque catégorie
#print(num_data_red['fixed acidity'].value_counts())


print(all_data_red)
#print(fixed_acidity)

all_data_red.info()
print(all_data_red.describe())#to get all the statistic informations
print(all_data_red.isnull().sum())#count the null value of all dataset fields

num_data_white.info()
print(num_data_white.describe())
print(num_data_white.groupby('quality').mean())#to look at all field inluences on quality value and find average values for each parameters

my.bar_plot_discrete_variables(num_data_white['quality'], 'White wine : quality', 'quality')
my.bar_plot_discrete_variables(num_data_red['quality'], 'Red wine : quality', 'quality')
my.plot_Categorial_distribution(categorized_acidity, 'categorized acidity')
my.plot_Categorial_distribution(categorized_quality, 'categorized quality')
my.plot_boxplot_histogram_qqplot(volatile_acidity, 'volatile acidity', 'volatile acidity', 'Count')

# matrice 12 par 12
M_corr_red = num_data_red.corr()
my.heatMap(M_corr_red, 'Vin rouge')#print correlation matrice

M_corr_white = num_data_white.corr()
my.heatMap(M_corr_white, 'Vin blanc')

#-------- TABLE DE L'ANOVA --------#
model = ols('alcohol ~ categorized_quality', data=all_data_red).fit()# Définir le modèle
anova_table = sm.stats.anova_lm(model, typ=2)# Effectuer l'ANOVA
print(anova_table)
A = all_data_red[all_data_red['categorized quality'] == 'good']['alcohol']
B = all_data_red[all_data_red['categorized quality'] == 'medium']['alcohol']
C = all_data_red[all_data_red['categorized quality'] == 'bad']['alcohol']
f_value, p_value = stats.f_oneway(A, B, C)
print(f'f_value : {f_value:0.4}, p_value : {p_value:0.4}')
# Interprétation de l'ANOVA
alpha = 0.05
if p_value < alpha:
    print("Au risque de 5%, la concentration moyenne en alchool diffère significativement entre les vins des 3 indexs de qualité.")
else:
    print("Au risque de 5%, il n'y a aucune différence significative des concentrations moyennes en alchool entre les vins des 3 indexs de qualité.")

for i in num_data_red.items():
    my.plot_boxplot_histogram_qqplot(i[1], 'Red wine : '+ i[0], i[0], 'Count')
    
for i in num_data_white.items():
    my.plot_boxplot_histogram_qqplot(i[1], 'White wine : '+ i[0], i[0], 'Count')