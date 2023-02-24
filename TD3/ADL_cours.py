#%%
from sklearn import datasets
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

X['Group'] = iris.target
X['Group'] = X['Group'].astype("category")

n = X.shape[0]
p = X.shape[1]

#%%
## 2
X.groupby('Group').mean()
X.groupby('Group').var()
X.boxplot(by='Group')
plt.suptitle("Répartition des mesures")
plt.show
#%%
## 3
pd.plotting.scatter_matrix(X.iloc[:,0:4], c=X['Group'])
plt.suptitle("Nuage de points 2 à 2")
plt.show()

# %%
## 4
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
coord_lda = lda.fit_transform(X.iloc[:,0:4], X['Group'])

plt.scatter(
    coord_lda[:,0], coord_lda[:,1],
    c=X['Group'], label=iris.target_names
)
plt.legend()
plt.xlabel("Composante discriminante 1")
plt.ylabel("Composante discriminante 2")
plt.title("Individus dans le plan des CP1 et CP2")
plt.show()

# %%
## 5
# corvar est de dimension (n,2) : contient dans la colonne 0 : la corrélation entre la composante principale 1 et les variables de départ 
# et dans la colonne 1 la corrélation entre la composante principale 2 et les variables de départ

corvar = np.zeros((p,2))

for i in range(p):
    for j in range(2):
        corvar[i,j] = np.corrcoef(X.iloc[:,i], coord_lda[:,j])[0,1]

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
plt.xlabel("Composante discriminante 1")
plt.ylabel("Composante discriminante 2")
plt.title('Cercle des corrélations')
plt.scatter(corvar[:,0],corvar[:,1])
#affichage des étiquettes (noms des variables)
for j in range(p):
  plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

plt.show()

# %%
# Prochaine fois : lda.predict
# Utiliser pour faire deviner à Python les variétés de fleurs
# chercher matrice de confusion