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


n = X.shape[0]
p = X.shape[1]

#%%
## 1
td.printMeanAndVar(X)
td.displayBoxplot(X)

#%%
## 2
td.displayCorrelationMatrix(X, "Correlation matrix of the data")
td.displayScatterMatrix(X, "Scatter matrix of the data")

#%%
## 3
td.displayCovarianceMatrix(X, "Covariance matrix of the data")

#%%
## 4
# from sklearn.decomposition import PCA
# acp = PCA()
# from sklearn.preprocessing import scale

# Xcr = scale(X, with_mean=True, with_std=True)
# Xacp = acp.fit_transform(Xcr)

#%%
## 5

td.displayParetoDiagram(X, "Pareto graph for the centered data", True)

# Ici on voit qu'avex deux composantes principales, on s'approche suffisamment
# des 100% de la variance cumulée donc elles suffiront pour décrire

# %%
## 6

print(td.totalVariance(X, True))

# %%
## 7

td.displayPopulationInFirstMainComponents(X, True)

# %%
## 8

td.displayCorrelationCircle(X, True)

# %%
