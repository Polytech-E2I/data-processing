#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import os.path as op

# Get the absolute path to the dataset that is alongside the Python script
scriptdir = op.dirname(op.abspath(__file__))
datafile = op.join(scriptdir, "Exercices.xlsx")

# Read dataset
X = pd.read_excel(
    datafile,
    sheet_name="Feuil1",    # Name of the sheet to use
    header=0,               # Row to use as column labels
    index_col=0             # Column to use as row labels
)
# Drop missing values
X.dropna()
# Indicate which columns are categories instead of values
categories_list = ['Gender', 'Smokes', 'Alcohol', 'Exercise', 'Ran']
X[categories_list] = X[categories_list].astype("category")
# Display information
# pprint(X.describe())
# pprint(X.info())

# Separate categories and numerical data
Xnum = X.select_dtypes(exclude=['category'])
Xcat = X.select_dtypes(include=['category'])

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
# pprint(Xnum)
# pprint(Xcat)

# Mean and variance
# pprint(Xnum.mean())
# pprint(Xnum.var())



# Boxplot
Xnum.boxplot()
plt.ylabel("Valeurs")
plt.title("Répartition des différentes valeurs")
# Histogram of quantitative variables
Xnum[['Height', 'Weight']].hist()
plt.ylabel("Quantité absolue")
plt.suptitle("Répartition des tailles et poids")
Xnum[['Pulse1', 'Pulse2']].hist()
plt.ylabel("Quantité absolue")
plt.suptitle("Répartition des pouls relevés")

plt.figure()
ax1 = plt.subplot(1, 2, 1)
Xnum['Age'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel("Quantité absolue")
ax2 = plt.subplot(1, 2, 2)
Xnum['Year'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel("Quantité absolue")
plt.suptitle("Répartition des âges et promotions")

plt.show()

