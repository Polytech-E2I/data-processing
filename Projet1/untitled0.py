# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:59:01 2023

@author: guyadern
"""
# Pour charger des données au format .mat
from sklearn.decomposition import PCA
from sklearn import preprocessing
import scipy.io

data = scipy.io.loadmat('data.mat')

X1 = data['data1']
X2 = data['data2']
X3 = data['data3']

# Pour centrer et réduire des données
scaler = preprocessing.StandardScaler()
# Rappel de quelques fonctions utilisées en TrD

# Classe pour l'ACP
# Instanciation
acp = PCA()
# affichage des paramètres
print(acp)
# On récupère les coordonnés des observations dans le nouvel espace
coord = acp.fit_transform(X1)  # variance expliquée
print(acp.explained_variance_ratio_)
