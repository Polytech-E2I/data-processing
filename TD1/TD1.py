#!/usr/bin/python

#%%
# Import modules

import sys
import os.path as op
libdir = op.join(op.dirname(op.dirname(op.abspath(__file__))), "TDlib")
sys.path.append(libdir)
from TDlib import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

#%%
# Import data

X = excelToPandas(
    "Exercices.xlsx",
    "Feuil1",
    0,
    0,
    ['Gender', 'Smokes', 'Alcohol', 'Exercise', 'Ran']
)
Xcat = separateNumAndCat(X)['Xcat']
Xnum = separateNumAndCat(X)['Xnum']

#%%
# Change categories to their explicit meanings
categories_values = {
    "Gender": {1: 'Male', 2: 'Female'},
    "Smokes": {1: "Smoker", 2: "Non-smoker"},
    "Alcohol": {1: "Drinks", 2: "Doesn't drink"},
    "Exercise": {1: "High", 2: "Moderate", 3: "Low"},
    "Ran":      {1: "Ran", 2: "Sat"}
}
for i in categories_values.keys():
    Xcat[i].replace(categories_values[i], inplace=True)

#%%
# Display Dataframes
prettyPrintDataframe(X)

#%%
# Display mean and variance
printMeanAndVar(X)

#%%
# Boxplot
displayBoxplot(X)

#%%
# Histogram of quantitative variables
displayDualHistograms(
    X,
    ['Height', 'Weight'],
    ("Height (cm)", "Weight (kg)"),
    "Spread of heights and weights"
)
displayDualHistograms(
    X,
    ['Pulse1', 'Pulse2'],
    ("Pulse1 (bpm)", "Pulse2 (bpm)"),
    "Spread of pulses"
)
######## HISTOGRAMMES POUR QUANTITA, BARGRAPH FOR QUALITA ######################

#%%
###### NP.UNIQUE PERMET DE FAIRE APPARAITRES LES ZÉROS #########################
plt.figure()
ax1 = plt.subplot(1, 2, 1)
#Xnum['Age'].value_counts().plot(kind='bar', ax=ax1)
values, counts = np.unique(Xnum.iloc[:, 2], return_counts=True)
ax1.bar(values, counts, width=0.5)
ax1.set_ylabel("Quantité absolue")
ax2 = plt.subplot(1, 2, 2)
Xnum['Year'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel("Quantité absolue")
plt.suptitle("Répartition des âges et promotions")

#plt.show()

print("Modalités pour Xcat :\n", Xcat.value_counts())

Xnum.plot.scatter(x='Height', y='Weight')
pd.plotting.scatter_matrix(Xnum)

pprint(Xnum.corr())

crosstab = pd.crosstab(X['Gender'], X['Smokes'])
crosstab.plot(kind='bar', stacked=True)

pprint(X.groupby("Gender").mean())

X.boxplot(column=['Height'], by=['Gender'])

plt.show()
