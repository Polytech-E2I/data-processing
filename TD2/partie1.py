#!/usr/bin/python

#%%
# Import modules
import sys
import os.path as op
libdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(libdir)
import TDlib as td

#%%
# Import data

X = td.excelToPandas(
    "Notes.xlsx",
    0,
    0,
    0,
    []
)

#%%
## 1. Réaliser une analyse univariée des variables (boxplot, moyenne et variance de chaque variable)
td.printMeanAndVar(X)
td.displayBoxplot(X)

#%%
## 2. Réaliser une analyse bivariée des variables (graphiques des variables 2 à 2, calcul de la matrice des corrélations)
td.displayCorrelationMatrix(X, "Correlation matrix of the grades")
td.displayScatterMatrix(X, "Scatter matrix of the grades")

#%%
## 3. Donner la matrice de variance-covariance
td.displayCovarianceMatrix(X, "Covariance matrix of the grades")

#%%
## 4. Réaliser l’ACP en utilisant la fonction
# from sklearn.decomposition import PCA
# acp = PCA()
# Xacp = acp.fit(X).transform(X)
# print(Xacp)

#%%
## 5. Présenter les variances expliquées par chaque axe (ou les pourcentages de variance) sous la forme d’un pareto (voir cours)
td.displayParetoDiagram(X, "Pareto diagram of the grades")

# Ici on voit qu'avex deux composantes principales, on s'approche suffisamment
# des 100% de la variance cumulée donc elles suffiront pour décrire

# %%
## 6. Donnez la variance totale du nuage de points-individus ? (de 2 manières)

print(td.totalVariance(X))

# %%
## 7. Donner la figure qui présentent les individus dans l’espace des 2 composantes principales.
td.displayPopulationInFirstMainComponents(X)

# %%
## 8. Pour interpréter les composantes principales on va présenter la figure : cercle des corrélations. Créer une matrice corvar de taille (4, 2) contenant les corrélations entre les 4 variables de départ et les 2 composantes principales (np.corrcoef)

td.displayCorrelationCircle(X)

# %%
