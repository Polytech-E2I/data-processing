#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()

#On peut ensuite se ramener à une DataFrame
Xdf = pd.DataFrame(iris.data,columns=iris.feature_names)
Xdf['target']=iris.target
names = iris.target_names

for i in range(len(iris['target'])):
    Xdf['names'] = names[iris['target']]

#%% pretraitement des données
Xdf[["target", "names"]] = Xdf[["target", "names"]].astype("category")
T2 = Xdf.describe()
Xdf = Xdf.dropna()

plt.figure()
Xdf.boxplot()
plt.show()

plt.figure()
Xdf.boxplot(by="names")
plt.show()

print("la moyenne est :\n", Xdf.groupby("target").mean())
print("la variance est :\n", Xdf.groupby("target").var())
# %%

pd.plotting.scatter_matrix(Xdf.iloc[:, 0:4], c=Xdf['target'])
# %% Analyse multivariée

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
coord_lda = lda.fit_transform(Xdf.iloc[:,0:4],Xdf['target'])

plt.figure()
plt.scatter(x = coord_lda[:,0], y = coord_lda[:,1], c=Xdf['target'])
plt.show()


# %%
def cercle(X, coord_lda) :
    p = Xdf.shape[1] - 2
    corvar = np.zeros((p, 2))

    for i in range(p) :
        for j in range(2) :
            corvar[i, j] = np.corrcoef(X.iloc[:,i], coord_lda[:,j]) [0, 1]

    # corvar est de dimension (n,2) : contient dans la colonne 0 : la corrélation entre la composante principale 1 et les variables de départ 
    # et dans la colonne 1 la corrélation entre la composante principale 2 et les variables de départ

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

cercle(Xdf, coord_lda)

# %%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

true = Xdf["target"]
predict = lda.predict(Xdf.iloc[:,0:4])

conf_mat = confusion_matrix(true, predict) # premier argument ligne, deuxieme argument colonne
print(conf_mat)
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()

conf_mat_norm = confusion_matrix(true, predict, normalize='true') # premier argument ligne, deuxieme argument colonne
print(conf_mat_norm)
disp = ConfusionMatrixDisplay(conf_mat_norm)
disp.plot()

conf_mat_norm_all = confusion_matrix(true, predict, normalize='all') # premier argument ligne, deuxieme argument colonne
conf_mat_norm_pred = confusion_matrix(true, predict, normalize='pred') # premier argument ligne, deuxieme argument colonne
print(conf_mat_norm_all)
print(conf_mat_norm_pred)

# %%
