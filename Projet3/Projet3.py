#%%
# Setup

import pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pprint
import sys
sys.path.append('..')
import TDlib as td

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
print("\nRépartition des classes :")
print(data["Classe"].value_counts(normalize=True) * 100)
print("\nRépartition des sous-classes :")
print(td.renameCategories(data, categories)["SousClasse"].value_counts(normalize=True) * 100)

#%%
# Descriptive stats

td.displayBoxplot(td.renameCategories(data, categories), "SousClasse")

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
# ACP

td.displayPopulationInFirstMainComponents(
    X_train.loc[:, X_train.columns!="SousClasse"],
    predict_column,
    ["Naturel", "Artificiel"]
)

# %%
# ALD

td.displayPopulationInFirstAndRandomDiscriminantComponents(
    X_train.loc[:, X_train.columns!="SousClasse"],
    predict_column,
    ["Naturel", "Artificiel"]
)

# %%
