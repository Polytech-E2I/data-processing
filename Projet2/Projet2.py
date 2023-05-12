#%%
# Setup

import os.path as op
import sys
libdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(libdir)
import TDlib as td
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

#%%
# Import data

data = pd.read_csv('dataset.csv')

categories = {
    "sex":      {0: "Female", 1: "Male"},
    "cp":       {1: "Typical angina", 2: "Atypical angina", 3: "non-aginal pain", 4: "asymptomatic"},
    "fbs":      {0: "Below", 1: "Above"},
    "restecg":  {0: "Normal", 1: "Abnormal"},
    "exang":    {0: "No", 1: "Yes"},
    "slope":    {1: "Upsloping", 2: "flat", 3: "Downsloping"},
    "thal":     {3: "Normal", 6: "Fixed defect", 7: "Reversable defect"},
    "target":   {0: "Healthy", 1: "Sick"}
}

for key in categories.keys():
    data[key] = data[key].astype("category")

column = "target"

#%%
# Describe data

print(data.describe())

#%%
# Display boxplots

td.displayBoxplot(
    td.renameCategories(data, categories),
    column,
    sharey=False
)

# %%
# DTC

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = data.loc[:, data.columns!=column]
Y = data[column]

accuracies = []

for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    dtc = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10
    )
    dtc.fit(x_train, y_train)

    true = y_test
    predict = dtc.predict(x_test)
    accuracy = accuracy_score(true, predict)

    # fig, ax = plt.subplots()
    # _ = plot_tree(
    #     dtc,
    #     ax=ax
    # )
    # plt.title(f"Arbre de décision, Accuracy = {accuracy:.2f}")
    # plt.show()

    accuracies.append(accuracy)

print(pd.DataFrame(accuracies).describe())

#%%
# Jouer sur :
# max_depth, min_samples_leaf, criterion, ccp_alpha