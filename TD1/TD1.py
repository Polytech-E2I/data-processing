#!/usr/bin/python

#%%
# Import modules

import sys
import os.path as op
libdir = op.join(op.dirname(op.dirname(op.abspath(__file__))), "TDlib")
sys.path.append(libdir)
import TDlib as td

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

#%%
# Import data

X = td.excelToPandas(
    "Exercices.xlsx",
    "Feuil1",
    0,
    0,
    ['Gender', 'Smokes', 'Alcohol', 'Exercise', 'Ran']
)
Xunits = {
    "Height":   "cm",
    "Weight":   "kg",
    "Age":      "years",
    "Pulse1":   "BPM",
    "Pulse2":   "BPM",
    "Year":     "from 1900"
}

Xcat = td.separateNumAndCat(X)['Xcat']
Xnum = td.separateNumAndCat(X)['Xnum']

#%%
# Display Dataframes
td.prettyPrintDataframe(X)

#%%
# Display mean and variance
td.printMeanAndVar(X)

#%%
# Boxplot
td.displayBoxplot(X)

#%%
# Histograms of quantitative numerical variables
td.displayHistograms(
    X,
    Xunits,
    ['Height', 'Weight'],
    "Spread of heights and weights"
)
td.displayHistograms(
    X,
    Xunits,
    ['Pulse1', 'Pulse2'],
    "Spread of pulses"
)
######## HISTOGRAMMES POUR QUANTITA, BARGRAPH FOR QUALITA ######################

#%%
# Bargraphs of qualitative numerical variables

td.displayBarGraphs(
    X,
    Xunits,
    ['Age', 'Year'],
    'Spread of ages and years'
)
#%%

# Change categories to their explicit meanings
categories_values = {
    "Gender":   {1: 'Male',     2: 'Female'},
    "Smokes":   {1: "Smoker",   2: "Non-smoker"},
    "Alcohol":  {1: "Drinks",   2: "Doesn't drink"},
    "Exercise": {1: "High",     2: "Moderate",          3: "Low"},
    "Ran":      {1: "Ran",      2: "Sat"}
}

temp = td.renameCategories(Xcat, categories_values)

print("DataFrame with renamed categories :\n", temp)

#%%
# Display scatter plot
td.displayTwoColumnScatter(
    X,
    Xunits,
    ("Height", "Weight"),
    "Weight against Height"
)

#%%
#pd.plotting.scatter_matrix(Xnum)
#%%
pprint(Xnum.corr())

#%%
crosstab = pd.crosstab(X['Gender'], X['Smokes'])
crosstab.plot(kind='bar', stacked=True)

#%%
pprint(X.groupby("Gender").mean())

X.boxplot(column=['Height'], by=['Gender'])

#%%
plt.show()
