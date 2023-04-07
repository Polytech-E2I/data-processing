#%%
import os.path as op
import sys
libdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(libdir)
import scipy.io
import pandas as pd
from pprint import pprint
import TDlib as td

data = scipy.io.loadmat('data.mat')

#%%

# data est un dictionnaire contenant 3 jeux de données data1, data2, data3
# data1 est un tableau contenant un tableau par forme d'onde
# Chacun de ces tableaux contient 9 paramètres extraits de la forme d'onde

data1 = data['data1']
data1_filtered = data1[:, 4:] # Seuls les paramètres 5-9 nous intéressent
data1_class    = data1[:, 1]  # Le diagnostic du médecin servira de classification

X1 = pd.DataFrame(
    data1_filtered,
    columns=["MaxInter", "DiffMax", "DiffMin", "DiffPress", "DiffWidth"]
)
X1["Valid"] = pd.Series(data1_class).astype("category")

categories = {
    "Valid": {1: "Artefact", 2: "True pulse"}
}

print(data1_class)
pprint(X1)


#%%
td.displayBoxplot(td.renameCategories(X1, categories), "Valid", False)

#%%
td.displayParetoDiagram(X1, "Pareto diagram")

print("On observe que deux composantes permettent de conserver environ 70% de l'information")


# %%
td.displayPopulationInFirstMainComponents(X1, "Valid", ["Artefact", "True pulse"])

print("Les groupes sont relativements distincts et bien \"patatoïdes\", on peut donc en apprendre des Gaussiennes")


#%%
td.displayConfusionMatrix(X1, "Valid")

print("On observe que la prédiction LDA est capable de prédire de manière assez fiable si un pouls est correct. Pour les artefact, c'est moins évident.")

# %%



# nb = GaussianNB()
# nb.fit(data1[colonnes de data], data1[états connus])
# nb.predict(data2[colonnes de data])

# => Matrice de confusion entre nb.predict et data2[états connnus]

# => Refaire avec les autres classifiers + autres datas
# ====> POSER LES TESTS CAR IL Y EN A BCP