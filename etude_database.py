import pandas as pd #dataframe with mixed types
import numpy as np #array with same type elements
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
import scipy.stats as stats

import stats_fct

#variables continues (age) boxplot, histogrammes, qqplot
#variables discretes (enfants) diagrammes en batons
#variables catégorielles (urbaine) camembert, diagramme à bandole

sns.set_theme()#set seaborn_theme()

Data = pd.read_csv("winequality-red.csv", sep=';', decimal='.')#load data form csv file

#define data columns as variables for easy access
columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide", "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

#extraction de colonnes par identifiant
fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality = (Data[col] for col in columns)

print(Data)

#creation d'une variable categorielle à partir d'une variable de float
# Définition d'une fonction pour mapper les valeurs de la colonne à "acide" ou "basic" en fonction de la condition
def map_acidity(value):
    if value < 7:
        return "acide"
    else:
        return "basic"

# Appliquer la fonction à la colonne "fixed acidity"
Data['fixed acidity'] = Data['fixed acidity'].apply(map_acidity)
fixed_acidity = Data['fixed acidity']
# Affichage du nombre d'occurrences de chaque catégorie
#print(Data['fixed acidity'].value_counts())


print(Data)
#print(fixed_acidity)

stats_fct.bar_plot_discrete_variables(quality, 'quality')
stats_fct.plot_Categorial_distribution(fixed_acidity, 'fixed acidity')
stats_fct.plot_boxplot_histogram_qqplot(volatile_acidity, 'volatile acidity', 'volatile acidity', 'Count')