#!/usr/bin/python

#%%
# Import modules
import sys
import os.path as op
libdir = op.join(op.dirname(op.dirname(op.abspath(__file__))), "TDlib")
sys.path.append(libdir)
import TDlib as td

import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import seaborn
import numpy as np

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
## 1
td.printMeanAndVar(X)

#%%
td.displayBoxplot(X)

#%%
## 2
td.displayCorrelationMatrix(X, "Correlation matrix of the grades")

#%%
td.displayScatterMatrix(X, "Scatter matrix of the grades")

#%%
## 3
td.displayCovarianceMatrix(X, "Covariance matrix of the grades")

#%%
## 4
from sklearn.decomposition import PCA
acp = PCA()
Xacp = acp.fit(X).transform(X)
print(Xacp)

#%%
# ## 5
td.displayParetoDiagram(X, "Pareto diagram of the grades")

# Ici on voit qu'avex deux composantes principales, on s'approche suffisamment
# des 100% de la variance cumulée donc elles suffiront pour décrire

# %%
## 6

# print(acp.explained_variance_.sum())
# print(X.var().sum())

# %%
## 7
td.displayPopulationInFirstMainComponents(X)

# %%
## 8

td.displayCorrelationCircle(X)

# %%
