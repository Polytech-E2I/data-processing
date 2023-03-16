#!/usr/bin/python

#%%
# Import modules
import sys
import os.path as op
libdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(libdir)
import TDlib as td
import pandas as pd

# %%
# Import data

from sklearn import datasets

iris = datasets.load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

X['Group'] = iris.target
X['Group'] = X['Group'].astype("category")

n = X.shape[0]
p = X.shape[1]

#%%
## 2

td.printMeanAndVar(X, "Group")

td.displayBoxplot(X, "Group")

#%%
## 3

td.displayScatterMatrix(X, "Scatter matrix grouped by species", "Group")

# %%
## 4

td.displayPopulationInFirstDiscriminantComponents(X, "Group", iris.target_names)

# %%
## 5

td.displayLDACorrelationCircle(X, 'Group')

# %%
# Prochaine fois : lda.predict
# Utiliser pour faire deviner à Python les variétés de fleurs
# chercher matrice de confusion