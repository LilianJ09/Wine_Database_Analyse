import pandas as pd #dataframe with mixed types
import numpy as np #array with same type elements
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
import scipy.stats as stats
import math

#variables continues (age) boxplot, histogrammes, qqplot
#variables discretes (enfants) diagrammes en batons
#variables catégorielles (urbaine) camembert, diagramme à bandole

sns.set_theme()#set seaborn_theme()

Data = pd.read_csv("Abena_Data.csv", sep=';', decimal=',')#load data form csv file

#define data columns as variables for easy access
columns = ["urbaine","age","couple","enfants","scolaire","situation","repas","duree","assurance","imc"]

#extraction de colonnes par identifiant
urbaine, age, couple, enfants, scolaire, situation, repas, duree, assurance, imc = (Data[col] for col in columns)


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
    plt.xlabel(xlabel)
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
    
''' 
plot_boxplot_histogram_qqplot(age, 'Age', 'Age', 'Count')
plot_boxplot_histogram_qqplot(duree, 'Duree', 'Duree', 'Count')
plot_boxplot_histogram_qqplot(imc, 'IMC', 'IMC', 'IMC')
plot_Categorial_distribution(urbaine, 'Urbaine')
plot_Categorial_distribution(couple, 'Couple')
plot_Categorial_distribution(scolaire, 'Scolaire')
plot_Categorial_distribution(situation, 'Situation')
plot_Categorial_distribution(assurance, 'Assurance')
bar_plot_discrete_variables(enfants, 'enfants')
'''

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
    

#test de shapiro wilk permet d'identifier si une variable continue est distribuée suivant une loi normale (alternative aux qqplots)
test_statistic, p_value = shapiro_wilk_test(age)#call the function
print(f'\nAge : Shapiro-Wilk Test statistic: {test_statistic:0.3}, p_value: {p_value:0.3}')
test_statistic, p_value = shapiro_wilk_test(duree)#call the function
print(f'\nDuree : Shapiro-Wilk Test statistic: {test_statistic:0.3}, p_value: {p_value:0.3}')
test_statistic, p_value = shapiro_wilk_test(imc)#call the function
print(f'\nIMC : Shapiro-Wilk Test statistic: {test_statistic:0.3}, p_value: {p_value:0.3}')


#synthese chiffree
column_num = ["age","enfants", "duree", "imc"]
data_num = Data[column_num]
data_num.describe()
m = data_num.mean()#esperance ou moyenne
s = data_num.std#ecart type (corrigé) div par N-1
# Créer un objet Series de Pandas à partir de vos données
#data_series = pd.Series(data_num)
# Calculer l'écart type en utilisant la méthode std()
#s = data_series.std(ddof=1)  # ddof=1 pour utiliser n-1 dans le dénominateur
data_num.min()
data_num.max()
data_num.quantile(0.25)
data_num.quantile(0.5)
data_num.quantile(0.75)

N = len(data_num)#nombre de donnees de la serie
alpha = 0.05#risque de 1er ordre
#test sur la moyenne, variance de la population inconnue donc test T student
#one sample t-test, t_statistic correspond au delta_observe
#This is a test for the null hypothesis that the expected value (mean) of a sample of independent observations a is equal to the given population mean, popmean.
#testing if the mean is significantly different from m0 = 45 years old
Mean_norme = 45#age moyen des femmes qui y ont recours < 45ans
#alternative "two-sided" (par défaut if H0 : m = m0), "less" (H0 : m < m0), ou "greater" (H0 : m > m0)
#t_statistic is the Delta obs
t_statistic, p_value = stats.ttest_1samp(age, popmean=Mean_norme, alternative = 'two-sided')#test de conformité à une moyenne
print(f'\nttest_1samp Age : test_statistic : {t_statistic:0.4}, p_value: {p_value:0.4}')
Delta_obs = (age.mean()-Mean_norme)/(age.std()/math.sqrt(len(age)))#verify
print(f'Delta_obs {Delta_obs}')
#si p_value < alpha on rejette l hypothèse 0 selon laquelle la moyenne de l'échantillon est égale à la moyenne spécifiée
if(p_value<alpha):
    print(f'H0 rejected -> gap with mean norme is significant m != {Mean_norme}')
else:
    print(f'H0 non rejected -> m = {Mean_norme}')
    
    
Mean_norme = 2.5#duree de recours à l'aide < 2.5 mois
t_statistic, p_value = stats.ttest_1samp(duree, popmean=Mean_norme, alternative = 'two-sided')#test de conformité à une moyenne
print(f'\nttest_1samp Duree : test_statistic : {t_statistic:0.3}, p_value: {p_value:0.3}')
#si p_value < alpha on rejette l hypothèse 0 selon laquelle la moyenne de l'échantillon est égale à la moyenne spécifiée
if(p_value<alpha):
    print(f'H0 rejected -> gap with mean norme is significant m != {Mean_norme}')
else:
    print(f'H0 non rejected -> m = {Mean_norme}')
    

Mean_norme = 25#'IMC moyen des femmes qui ont recours à l'aide alimentaire est supérieur a 25
#alternative = 'greater' : test unilatéral pour déterminer si la moyenne de l'échantillon est significativement plus grande que la valeur spécifiée
t_statistic, p_value = stats.ttest_1samp(imc, popmean=Mean_norme, alternative = 'greater')#test de conformité à une moyenne
print(f'\nttest_1samp IMC : test_statistic : {t_statistic:0.3}, p_value: {p_value:0.3}')
# Si p_value < alpha, on rejette l'hypothèse nulle selon laquelle la moyenne de l'échantillon est égale à la moyenne spécifiée
if p_value < alpha:
    print(f'H0 rejected -> la moyenne IMC est significativement plus grande que {Mean_norme}')
else:
    print(f'H0 non rejected -> la moyenne IMC est <= {Mean_norme}')
    
    
#two sample t-test
#filtrer les ages en 2 groupes : seule vs couple
#filter to get ages of women living alone
age_seule = Data[Data['couple'] == 'seule']['age']
#filter to get ages of women living in a couple
age_couple = Data[Data['couple'] == 'couple']['age']

#test de Levene d'égalité des variances (moins sensibles à l'écart de normalité que Bartelett par exemple)
stat, p_value = stats.levene(age_seule, age_couple)
print(f"\nLevene's test statistic: {stat:.3f}, P_value : {p_value:.3f}")

#testing if there's a significant difference between the mean of two independent groups
t_stat, p_value = stats.ttest_ind(age_seule, age_couple, equal_var = True, alternative = 'two-sided')
print("\nTwo-sample t-test statistic:", t_stat, "P_value:", p_value)


#Test d'association de 2 variables categorielles
from scipy.stats import chi2_contingency
#contingency table
contingency_table = pd.crosstab(repas, situation)
print("\nContingency table between 'repas' and 'situation' :")
print(contingency_table)

#chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square Test :\nChi2 Statistic : {chi2:4}, P-value : {p:4}")
print(expected)

#check if the result is statistically significant
if p<0.05:
    #calculate Cramer's V
    n = contingency_table.sum().sum()#total sample size
    phi2 = chi2/n
    v = np.sqrt(phi2/min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1))
    print(f"v : {v}")
    #adjusted interpretation based on degrees of freedom*
    def interpret_cramers_v(v, dof, n):
        #standardisartion pour que ce soit compris entre 0 et 1
        adjusted_v = v / np.sqrt(dof/n)
        if adjusted_v < 0.10:
            return 'Tres faible'
        elif adjusted_v < 0.20:
            return 'Faible'
        elif adjusted_v < 0.40:
            return 'Modérée'
        elif adjusted_v < 0.60:
            return 'Forte'
        elif adjusted_v < 0.80:
            return 'Tres forte'
        else:
            return 'Extrement forte'
    
    print(f"\nCramer's V : {v:.3f}")
    print(f"Adjusted interpretation based on dof ({dof}): {interpret_cramers_v(v, dof, n)}")
     
     
     
contingency_table = pd.crosstab(scolaire, situation)
print("\nContingency table between 'scolaire' and 'situation' :")
print(contingency_table)

#chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square Test :\nChi2 Statistic : {chi2:4}, P-value : {p:4}")
print(expected)

#check if the result is statistically significant
if p<0.05:
    #calculate Cramer's V
    n = contingency_table.sum().sum()#total sample size
    phi2 = chi2/n
    v = np.sqrt(phi2/min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1))
    print(f"v : {v}")
    #adjusted interpretation based on degrees of freedom*
    def interpret_cramers_v(v, dof, n):
        #standardisartion pour que ce soit compris entre 0 et 1
        adjusted_v = v / np.sqrt(dof/n)
        if adjusted_v < 0.10:
            return 'Tres faible'
        elif adjusted_v < 0.20:
            return 'Faible'
        elif adjusted_v < 0.40:
            return 'Modérée'
        elif adjusted_v < 0.60:
            return 'Forte'
        elif adjusted_v < 0.80:
            return 'Tres forte'
        else:
            return 'Extrement forte'
    
    print(f"\nCramer's V : {v:.3f}")
    print(f"Adjusted interpretation based on dof ({dof}): {interpret_cramers_v(v, dof, n)}")
    
    standardized_residuals = (contingency_table - expected)/np.sqrt(expected)
    
    #create heatmap of standardized residuals
    plt.figure(figsize=(10,5))
    sns.heatmap(standardized_residuals, annot=True, cmap='coolwarm', center=0)
    plt.title("Heatmap of Standardized Residuals")
    plt.ylabel('Situation')
    plt.xlabel('Scolaire')
    plt.show()
    
    #determine significant attraction and repulsions
    print("\nsignificant interaction")
    for index, row in standardized_residuals.iterrows():
        for col, value in row.items():
            if value > 1.96:
                print(f"Attraction between {index} and {col} (Residual : {value:.2f})")
            elif value < - 1.96:
                print(f"Repulsion between {index} and {col} (Residual: {value:.2f})")
    else:
        print("No significant association found between the variables")
        

#boites a moustaches : les encoches (notches) representent un intervalle de confiance rustique de la médiane
#lorsque la distribution est symetrique, la mediane et la moyenne sont superposees
#p valeur compatibilite entre l hypothese de l'echantillon et delta obs
######ANDVA
from statsmodels.formula.api import ols
from scipy.stats import levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#setting the aesthetic style for better visualisation
sns.set(style="whitegrid")

#creating boxplot
plt.figure(figsize=(10,6))#taille de la fig
sns.boxplot(x='situation', y='imc', data=Data, notch=True)
plt.title("Distribution de l'imc par situation professionnelle")
plt.ylabel('Situation')
plt.xlabel('Indice de masse corporelle (IMC)')
plt.show()

#realisation de l'ANOVA
model = ols('\nimc ~ C(situation)', data=Data).fit()#on met un c autour de situation pour
anova_results = sm.stats.anova_lm(model, typ=2)#typ=2 pour ANOVA type II
print(anova_results)

#verif signification de l'ANOVA
if anova_results['PR(>F)'][0] < 0.05:
    print("L'ANOVA est significative, procédons à un test de Tukey")
    #prepa des donnees pour le test
    tukey_results = pairwise_tukeyhsd(endog = Data['imc'], groups=Data['situation'], alpha=0.05)
    print(tukey_results)
    tukey_summary = tukey_results.summary()
    print(tukey_summary)
    #affichage graphique des résultats du test de Tukey
    fig = tukey_results.plot_simultaneous(comparison_name=Data['situation'].iloc[0], figsize=(10,6))
    plt.title('Comparaisons multiples - test de Tukey HSD')
    plt.show()
else:
    print("L'ANOVA n'est pas significative, pas besoin de test post-hoc")
    
####conditions de validité du modele
#calcul des residus du modele
residuals = model.resid
#verif de la normalite des residus
#graphiquement avec un qqplot
sm.qqplot(residuals, line='45', fit=True)
plt.title('Graphique Q-Q des résidus')
plt.show()
'''
Les points sur le graph qq plot doivent idealement suivre la ligne diagonale (y=x). Des deviations significatives
de cette ligne indiquent des ecarts par rapport à la normalié. Des courbures aux extremites peuvent indiquer des queues
lourdes ou legeres, ce qui est un signe de non-normalité. La normalite des residus est plus critique dans les petuts
echantillons. Dans les grands echantillons, selon le th. central limite, les petites déviations de la normalite sont svnt
moins problematiques
'''

#avec le test de shapiro-wilk
stat, p_value = shapiro(residuals)
print("statistique de shapiro-wilk:", stat)
print("P-value du test de shapiro-wilk:",p_value)

#verif de l'homoscédasticité dds résidus
#recup des valeurs prédites
fitted = model.fittedvalues
#graphiquement : residus vs valeurs predites
plt.figure(figsize=(8,6))
plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title("Graphique de Résidus vs. Valeurs prédites")
plt.show()
'''
Résidus vs. Valeurs prédites : On cherche une dispersion constante des points autour de la ligne centrale (zero)
Des motifs clairs ou une dispersion variable des residus peuvent indiquer des problemes d'homoscédasticité
'''
#par le test de Levene (comparaison de moyennes)
group_data = [Data['imc'][Data['scolaire'] == level] for level in Data['scolaire'].unique()]
stat, p = levene(*group_data)
print(f'Test de Levene : Statistique = {stat}, p-valeur = {p}')