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
    "Criminalite.xlsx",
    0,
    0,
    0,
    []
)

#%%
## 1. Observer très rapidement ces données et regarder si des états se différencient des autres etc.
td.printMeanAndVar(X)
td.displayBoxplot(X)

#%%
## 2. Donner les moyennes et les écarts types des différentes variables. Donnez également la variance totale du nuage de points.

td.displayCorrelationMatrix(X, "Correlation matrix of the data")
td.displayScatterMatrix(X, "Scatter matrix of the data")

#%%
## 3. Regarder les distributions des différentes variables.
td.displayCovarianceMatrix(X, "Covariance matrix of the data")

#%%
## 4. Calculer les corrélations entre les variables.
# from sklearn.decomposition import PCA
# acp = PCA()
# Xacp = acp.fit(X).transform(X)
# print(Xacp)

#%%
## 5. Utiliser une représentation graphique pour visualiser les coefficients de corrélations

td.displayParetoDiagram(X, "Pareto graph for the data")

# Ici on voit qu'avex deux composantes principales, on s'approche suffisamment
# des 100% de la variance cumulée donc elles suffiront pour décrire

# %%
## 6. Commentez les résultats

print(td.totalVariance(X))

# %%
## 7

td.displayPopulationInFirstMainComponents(X)

# %%
## 8

td.displayCorrelationCircle(X)

# %%
