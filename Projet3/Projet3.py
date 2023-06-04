#%%
# Setup

import pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pprint
import sys
sys.path.append('..')
import TDlib as td
import matplotlib.pyplot as plt

#%%
# Import data

data = pd.read_csv(
    'database.csv',
    header=0,
    index_col=0
)

data["Classe"]      = data["Classe"].astype('category')
data["SousClasse"]  = data["SousClasse"].astype('category')

inv_categories = {
    "Classe": {
        "NATUREL": 0,
        "ARTIFICIEL": 1
    },
    "SousClasse": {
        "COTE"          : 0,
        "FORET"         : 1,
        "AUTOROUTE"     : 2,
        "VILLE"         : 3,
        "MONTAGNE"      : 4,
        "OPEN_COUNTRY"  : 5,
        "RUE"           : 6,
        "GRANDBATIMENT" : 7
    }
}

categories = {}

for key in inv_categories.keys():
    data[key] = data[key].map(inv_categories[key])

    categories[key] = dict(
        zip(
            inv_categories[key].values(),
            inv_categories[key].keys()
        )
    )

predict_column = "Classe"

X = data
Y = data[predict_column]

#%%
# Describe data

# Nb individus
# Nb variables + type
# Nb individus par class

print(data.describe(percentiles=[]))
print(data.dropna().describe(percentiles=[]))
print("Écarts-type")
print(data.std())

td.renameCategories(data, categories).groupby('Classe').size().plot(kind='pie', autopct='%.2f %%')
plt.title("Répartition des classes")
plt.show()
td.renameCategories(data, categories).groupby('SousClasse').size().plot(kind='pie', autopct='%.2f %%')
plt.title("Répartition des sous-classes")
plt.show()

#%%
# Descriptive stats

td.displayBoxplot(td.renameCategories(data, categories), "Classe", sharey=False)

# %%
# Separate train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Re-diviser X_train en training et validation pour la phase de cross-validation
# Googler cross-validation PAS A LA MAIN
# Checker exemple dans cours
# Permet de tune kernel & param. C pour le SVM, ou alors le nb de neurones et la
# fct d'activation du MLP
# (tracer accuracies)

# Une fois qu'on a trouvé le meilleur SVM et le meilleur MLP, passer sur la base
# X_test pour les comparer

#%%
# ACP Classe

td.displayPopulationInFirstMainComponents(
    X_train.loc[:, X_train.columns!="SousClasse"],
    predict_column,
    ["Naturel", "Artificiel"]
)

# %%
# ALD Classe

td.displayPopulationInFirstAndRandomDiscriminantComponents(
    X_train.loc[:, X_train.columns!="SousClasse"],
    predict_column,
    ["Naturel", "Artificiel"]
)

# %%
# SVM

from sklearn.model_selection import cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

kernels = ["linear", "poly", "rbf", "sigmoid"]

X_val = td.separateNumAndCat(X_train)['Xnum']

for kernel in kernels :

    c_values = []
    accuracies = []

    for C in np.linspace(0.1, 10, 10):
        clf = svm.SVC(kernel=kernel, C=C, random_state=42)
        scores = cross_val_score(clf, X_val, Y_train, cv=5)

        print(f"{kernel}, C={C} : {scores.mean()*100:.2f} %")

        c_values.append(C)
        accuracies.append(scores.mean()*100)

    plt.plot(c_values, accuracies, label=kernel)

plt.xlabel("C value")
plt.ylabel("Accuracy score (%)")
plt.title(f"Accuracy en fonction de C")
plt.legend()
plt.show()

# %%
