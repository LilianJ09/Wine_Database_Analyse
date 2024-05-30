import pandas as pd #dataframe with mixed types
import numpy as np #array with same type elements
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
import scipy.stats as stats

#variables continues boxplot, histogrammes, qqplot
#variables discretes diagrammes en batons
#variables catégorielles camembert, diagramme à bandole

# Extracte the data base
def extract_red_Wine():
    return(pd.read_csv("../Bases de données/winequality-red.csv", sep=';', decimal='.'))

def extract_white_Wine():
    return(pd.read_csv("../Bases de données/winequality-white.csv", sep=';', decimal='.'))

# to create categorized variable
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

def plot_boxplot_histogram_qqplot(data, title, xlabel, ylabel):
    plt.figure(figsize=(12,6))
    # Create a boxplot and a historgam for age VA
    #VA age follow a normal law
    #legère dissimetrie gauche pour l'age, avantage de visualiser les données abérantes dans le box plott.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    #fonction boxplot de seaborn, si notch = false il fait des boites a moustaches
    #tout ce qui sort des moustaches est atypique
    sns.boxplot(y=data, color='lightblue', notch=True, flierprops={'marker':'o', 'markersize':8, 'markerfacecolor':'red'})#détecter les valeurs atypiques éventuelles
    plt.title(f'Boxplot of {title}')
    plt.subplot(1,3,2)
    sns.histplot(data, kde=False)
    plt.title(f'Histrogram of {title}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig=plt.subplot(1,3,3)
    #QQ plot pour voir si visuellement le jeu de données est proche d'une loi normale
    #si loi normale, les points s'alignent à la droite x y
    #confrontation des fractiles de la loi normale avec ceux du jeu de données
    sm.qqplot(data, ax=fig, line='s')#line standardiser for normale fractiles law
    #pour l'age, les données empiriques et théroqies sont bien ajustées, ecart seulemen,t sur les grandes valeurs
    #parametres défini de la loi normales sont ceux de notre jeu de données
    plt.title(f'QQ Plot of {title}')
    plt.suptitle(title, fontsize=15, y=0.97)  # Ajout du titre global
    plt.tight_layout()
    plt.show()
    
def plot_Categorial_distribution(data, title):
    frequencies = data.value_counts(normalize=True)
    fig, ax = plt.subplots(2,1,figsize=(12,6))
    ax[0].pie(frequencies, labels=frequencies.index, autopct='%1.1f%%')
    ax[0].set_title(f'Pie Chart of {title}')
    sns.barplot(x=frequencies.index, y=frequencies.values, ax=ax[1])
    ax[1].set_title(f'Bar Chart of {title}')
    plt.suptitle(f'Variable catégorielle : {title}', fontsize=15, y=0.97)  # Ajout du titre global
    plt.tight_layout()
    plt.show()
    

def bar_plot_discrete_variables(data, title, xlabel, ylabel = 'Frequency'):
    #diagramme en batons
    frequencies = data.value_counts(normalize=True)
    sns.barplot(x=frequencies.index, y=frequencies.values)
    plt.title(f'Bar plot : {ylabel} of {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.xticks(rotation=45)  # Rotation des étiquettes sur l'axe des x pour une meilleure lisibilité
    plt.suptitle(f'{title}', fontsize=15, y=0.97)  # Ajout du titre global
    plt.tight_layout()
    plt.show()
    

#test de normalité de shapiro-wilk
"""
perform the Shapiro-Wilk test for normality
Args : data (pd.Series): A pandas Series containing numeric data
Returns : float the N test statistic, float the p-value for the hypothesis test
"""
def shapiro_wilk_test(data):
    #remove NaN values which can't be handled by the Shapiro-wilk, enlever les données manquantes
    data_clean = data.dropna()
    
    #perfirming the shapiro wilk test
    stat, p_value = shapiro(data_clean)
    
    #interpreting the result
    alpha = 0.05
    if p_value>alpha:
        print('X looks Gaussian (fail to reject H0)')
    else:
        print('X does not look Gaussian (reject H0)')
        
    return stat, p_value
# Créer une heat map à partir d'une matrice
def heatMap(M, title:str, center=0, vmax=1, vmin=-1):
    plt.title(f'Table de corrélation {title}')
    # Tracer la heatmap avec la colormap personnalisée
    sns.heatmap(M, annot=False, cmap='coolwarm', center=center, vmax=vmax, vmin=vmin)
    
    # Ajouter les annotations pour chaque cellule de la heatmap
    for i in range(len(M)):
        for j in range(len(M)):
            plt.text(j + 0.5, i + 0.5, '{:.2f}'.format(M.iloc[i, j]),
                     ha='center', va='center', color='black', fontsize=9)
    plt.show()