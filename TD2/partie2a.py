#%%
import matplotlib.pyplot as plt
import pandas as pd
import os.path as op
from pprint import pprint
import seaborn
import numpy as np

# Get the absolute path to the dataset that is alongside the Python script
scriptdir = op.dirname(op.abspath(__file__))
datafile  = op.join(scriptdir, "Criminalite.xlsx")

# On charge les données
X = pd.read_excel(
    datafile,
    sheet_name=0,
    header=0,
    index_col=0
)

n = X.shape[0]
p = X.shape[1]

#%%
## 1
pprint(X.mean())
pprint(X.var())
X.boxplot()
plt.ylabel("Notes sur 20")
plt.title("Répartition des notes")

#%%
## 2
seaborn.heatmap(X.corr(), cmap="coolwarm", annot=True)
plt.show()
pd.plotting.scatter_matrix(X)
plt.title("Nuages de points 2 à 2")
plt.show()

#%%
## 3
print(X.cov())

#%%
## 4
from sklearn.decomposition import PCA
acp = PCA()
Xacp = acp.fit(X).transform(X)
print(Xacp)

#%%
## 5
y = list(acp.explained_variance_ratio_)
x = range(len(y))
ycum = np.cumsum(y)
plt.bar(x,y)
plt.plot(x,ycum,"-r")
plt.xlabel("Nombre de facteurs (ou vecteurs propres)")
plt.ylabel("Variances expliquées et cumul de variance expliquée")
plt.title("Diagramme de Pareto")

plt.show()

# Ici on voit qu'avex deux composantes principales, on s'approche suffisamment
# des 100% de la variance cumulée donc elles suffiront pour décrire
# %%
## 6

print(acp.explained_variance_.sum())
print(X.var().sum())

# %%
## 7
plt.scatter(Xacp[:, 0], Xacp[:, 1])
for i, label in enumerate(X.index):
    plt.annotate(label, (Xacp[i,0], Xacp[i,1]))

plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Individus dans le plan des CP1 et CP2")
plt.show()

# %%
## 8

# corvar est de dimension (n,2) : contient dans la colonne 0 : la corrélation entre la composante principale 1 et les variables de départ 
# et dans la colonne 1 la corrélation entre la composante principale 2 et les variables de départ

corvar = np.zeros((p,2))

for i in range(p):
    for j in range(2):
        corvar[i,j] = np.corrcoef(X.iloc[:,i], Xacp[:,j])[0,1]

# Cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

# On ajoute les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
# On ajoute un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title('Cercle des corrélations')
plt.scatter(corvar[:,0],corvar[:,1])
#affichage des étiquettes (noms des variables)
for j in range(p):
  plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

plt.show()
# %%
