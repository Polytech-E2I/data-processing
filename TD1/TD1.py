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
# Histograms of quantitative numerical variables
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
# Bargraphs of qualitative numerical variables

displayDualBarGraph(
    X,
    ['Age', 'Year'],
    ['Age (years)', 'Year (from 1900)'],
    'Spread of ages and years',
    True
)
#%%

print("Modalités pour Xcat :\n", Xcat.value_counts())

Xnum.plot.scatter(x='Height', y='Weight')
pd.plotting.scatter_matrix(Xnum)

pprint(Xnum.corr())

crosstab = pd.crosstab(X['Gender'], X['Smokes'])
crosstab.plot(kind='bar', stacked=True)

pprint(X.groupby("Gender").mean())

X.boxplot(column=['Height'], by=['Gender'])

plt.show()
