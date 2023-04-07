#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import datasets

X = pd.read_excel("INFARCTU.xlsx", index_col=0)

#%% pretraitement des données
X[["PRONO", "C"]] = X[["PRONO", "C"]].astype("category")
Xdf = X[X["C"] != 3]
Xinc = X[X["C"] == 3]

for i in range(2,9):
    plt.figure()
    Xdf.iloc[:, [0,i]].boxplot(by="C")
    plt.show()

print("la moyenne est :\n", Xdf.groupby("C").mean())
print("la variance est :\n", Xdf.groupby("C").var())
# %%
pd.plotting.scatter_matrix(Xdf.iloc[:, 2:9], c=Xdf["C"])
# %% Analyse multivariée

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
coord_lda = lda.fit_transform(Xdf.iloc[:,2:9],Xdf["C"])

rand = np.random.randn(int(np.shape(coord_lda)[0]))
plt.figure()
plt.scatter(x = coord_lda[:,0],y=rand, c=Xdf["C"])
plt.show()

# %%
def cercle(data,acp,pstart):
    n,p=data.shape
    corvar=np.zeros((p-pstart,2))

    for k in range(pstart,p) :
        corvar[k-pstart,0]=np.corrcoef(data.iloc[:,k],acp[:,0])[0,1]
    ##il faut comprendre cette ligne

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
    for j in range(pstart,p):
      plt.annotate(data.columns[j],(corvar[j-pstart,0],corvar[j-pstart,1]))
    plt.show()

cercle(Xdf,coord_lda,2)

# %%
from sklearn.metrics import roc_curve

plt.figure()
for i in range(2, 9) :
    FP, TP, TH = roc_curve(Xdf.iloc[: , 0], Xdf.iloc[:, i], pos_label=1)
    plt.plot(FP, TP, label=Xdf.columns[i])

plt.show()
# %%