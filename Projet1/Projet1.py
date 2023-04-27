#%%
# PYTHON SETUP
import os.path as op
import sys
libdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(libdir)
import scipy.io
import pandas as pd
from pprint import pprint
import TDlib as td

#%%
# DATA FORMATTING

# data est un dictionnaire contenant 3 jeux de données data1, data2, data3
# data1 est un tableau contenant un tableau par forme d'onde
# Chacun de ces tableaux contient 9 paramètres extraits de la forme d'onde

try:
    data = scipy.io.loadmat('data.mat')
except FileNotFoundError:
    data = scipy.io.loadmat('Projet1/data.mat')

data1 = data['data1']
data1_filtered = data1[:, 4:] # Seuls les paramètres 5-9 nous intéressent
data1_class    = data1[:, 1]  # Le diagnostic du médecin servira de classification
data2 = data['data2']
data2_filtered = data2[:, 4:] # Seuls les paramètres 5-9 nous intéressent
data2_class    = data2[:, 1]  # Le diagnostic du médecin servira de classification
data3 = data['data3']
data3_filtered = data3[:, 4:] # Seuls les paramètres 5-9 nous intéressent
data3_class    = data3[:, 1]  # Le diagnostic du médecin servira de classification

X1 = pd.DataFrame(
    data1_filtered,
    columns=["MaxInter", "DiffMax", "DiffMin", "DiffPress", "DiffWidth"]
)
X1["Valid"] = pd.Series(data1_class).astype("category")
X2 = pd.DataFrame(
    data2_filtered,
    columns=["MaxInter", "DiffMax", "DiffMin", "DiffPress", "DiffWidth"]
)
X2["Valid"] = pd.Series(data2_class).astype("category")
X3 = pd.DataFrame(
    data3_filtered,
    columns=["MaxInter", "DiffMax", "DiffMin", "DiffPress", "DiffWidth"]
)
X3["Valid"] = pd.Series(data3_class).astype("category")

categories = {
    "Valid": {1: "Artefact", 2: "True pulse"}
}

#%%
# DATA1 DESCRIBE
print("---- data1 ----")
nb_artef = len([artef for artef in data1_class if artef == 1])
nb_valid = len([valid for valid in data1_class if valid == 2])
print(f"Nombre d'artefacts : {nb_artef}")
print(f"Nombre de pouls : {nb_valid}")
print(X1.describe())

#%%
# DATA2 DESCRIBE
print("---- data2 ----")
nb_artef = len([artef for artef in data2_class if artef == 1])
nb_valid = len([valid for valid in data2_class if valid == 2])
print(f"Nombre d'artefacts : {nb_artef}")
print(f"Nombre de pouls : {nb_valid}")
print(X2.describe())

#%%
# DATA3 DESCRIBE
print("---- data3 ----")
nb_artef = len([artef for artef in data3_class if artef == 1])
nb_valid = len([valid for valid in data3_class if valid == 2])
print(f"Nombre d'artefacts : {nb_artef}")
print(f"Nombre de pouls : {nb_valid}")
print(X3.describe())

#%%
# DATA1 BOXPLOT
td.displayBoxplot(td.renameCategories(X1, categories), "Valid", False)

#%%
# DATA2 BOXPLOT
td.displayBoxplot(td.renameCategories(X2, categories), "Valid", False)

#%%
# DATA3 BOXPLOT
td.displayBoxplot(td.renameCategories(X3, categories), "Valid", False)

#%%
# CORRELATION MATRIX

td.displayCorrelationMatrix(X1, "Matrice de corrélation data1")

#%%
# SCATTER MATRIX

td.displayScatterMatrix(X1, "")

print("Aucune intercorrélation")

#%%
# PARETO DIAGRAM

td.displayParetoDiagram(X1, "Pareto diagram")
print("On observe que deux composantes permettent de conserver environ 70% de l'information")


# %%
# ACP

td.displayPopulationInFirstMainComponents(X1, "Valid", ["Artefact", "True pulse"])

print("Les groupes sont relativements distincts et bien \"patatoïdes\", on peut donc en apprendre des Gaussiennes")

#%%
# ACP CORRELATION CIRCLE
td.displayCorrelationCircle(X1)

#%%
# ALD

td.displayPopulationInFirstAndRandomDiscriminantComponents(
    X1,
    'Valid',
    ["Artefact", "True Pulse"]
)

#%%
# ALD CORRELATION CIRCLE

td.displayLDACorrelationCircle(X1, "Valid")

#%%
# FIT SPLIT-DATA1 TEST SPLIT-DATA1

column = 'Valid'

td.plot_roc_curves(
    X1,
    X1,
    column,
    "Courbes ROC\nFit : data1  /  Valid : data1"
)

td.displayConfusionMatrices(
    X1,
    X1,
    "Valid",
    "Fit : data1 / Valid : data1"

)

td.printConfusionMatrixScores(X1, X1, "Valid")

#%%
# FIT DATA1 TEST DATA2

td.plot_roc_curves(
    X1,
    X2,
    column,
    "Courbes ROC\nFit : data1  /  Valid : data2"
)

td.displayConfusionMatrices(
    X1,
    X2,
    "Valid",
    "Fit : data1 / Valid : data2"
)

td.printConfusionMatrixScores(X1, X2, "Valid")

#%%
# FIT DATA1 TEST DATA3

td.plot_roc_curves(
    X1,
    X3,
    column,
    "Courbes ROC\nFit : data1  /  Valid : data3"
)

td.displayConfusionMatrices(
    X1,
    X3,
    "Valid",
    "Fit : data1 / Valid : data3"
)

td.printConfusionMatrixScores(X1, X3, "Valid")

# %%

column = "Valid"

td.displayQDAConfusionMatrix(X1, X2, column, "QDA, Fit: data1 / Valid : data2")
td.displayQDAConfusionMatrix(X1, X3, column, "QDA, Fit: data1 / Valid : data3")
td.displayGNBConfusionMatrix(X1, X2, column, "GNB, Fit: data1 / Valid : data2")
td.displayGNBConfusionMatrix(X1, X3, column, "GNB, Fit: data1 / Valid : data3")

# %%
