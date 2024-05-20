import pandas as pd #dataframe with mixed types
import numpy as np #array with same type elements
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
import scipy.stats as stats

from matplotlib.colors import LinearSegmentedColormap

import stats_fct as my

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

#load data form csv file
num_data_red = pd.read_csv("winequality-red.csv", sep=';', decimal='.')

#load data form csv file
num_data_white = pd.read_csv("winequality-white.csv", sep=';', decimal='.')

#define data columns as variables for easy access
columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide", "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

#extraction de colonnes par identifiant
fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality = (num_data_red[col] for col in columns)

print(num_data_red)

# Appliquer la fonction à la colonne "fixed acidity"
all_data_red = num_data_red.copy()
all_data_red['categorized acidity'] = all_data_red['fixed acidity'].apply(map_acidity)
categorized_acidity = all_data_red['categorized acidity']
# Affichage du nombre d'occurrences de chaque catégorie
#print(num_data_red['fixed acidity'].value_counts())


print(all_data_red)
#print(fixed_acidity)

my.bar_plot_discrete_variables(quality, 'quality')
my.plot_Categorial_distribution(categorized_acidity, 'categorized acidity')
my.plot_boxplot_histogram_qqplot(volatile_acidity, 'volatile acidity', 'volatile acidity', 'Count')

# matrice 12 par 12
M_corr_red = num_data_red.corr()
my.heatMap(M_corr_red, 'Vin rouge')

M_corr_white = num_data_white.corr()
my.heatMap(M_corr_white, 'Vin blanc')