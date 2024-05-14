import statistic_fct

import pandas as pd #dataframe with mixed types
import numpy as np #array with same type elements
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
import scipy.stats as stats

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
#print(fixed_acidity)
    
statistic_fct.plot_boxplot_histogram_qqplot(fixed_acidity, 'Fixed acidity', 'Fixed acidity', 'Count')