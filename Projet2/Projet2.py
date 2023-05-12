#%%
# Setup

import os.path as op
import sys
libdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(libdir)
import TDlib as td
import pandas as pd
from pprint import pprint

#%%
# Import data

X = pd.read_csv('dataset.csv')

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
    X[key] = X[key].astype("category")

#%%
# Describe data

print(X.describe())

#%%
# Display boxplots

td.displayBoxplot(
    td.renameCategories(X, categories),
    "target",
    sharey=False
)

# %%
