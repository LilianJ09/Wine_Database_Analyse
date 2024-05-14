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
    plt.suptitle(f'Variable continue : {title}', fontsize=15, y=0.97)  # Ajout du titre global
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
    

def bar_plot_discrete_variables(data, title):
    #diagramme en batons
    frequencies = data.value_counts(normalize=True)
    sns.barplot(x=frequencies.index, y=frequencies.values)
    plt.title(f'Diagramme en bâtons nombre {title}')
    plt.xlabel(f'{title}')
    plt.ylabel('Frequency')
    #plt.xticks(rotation=45)  # Rotation des étiquettes sur l'axe des x pour une meilleure lisibilité
    plt.suptitle(f'Variables discrètes : {title}', fontsize=15, y=0.97)  # Ajout du titre global
    plt.tight_layout()
    plt.show()
    

#test de normalité de shapiro-wilk
def shapiro_wilk_test(data):
    """perform the Shapiro-Wilk test for normality
    parameters : data (pd.Series): A pandas Series containing numeric data
    Returns : float the N test statistic, float the p-value for the hypothesis test
    """
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