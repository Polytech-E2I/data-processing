#%%
# Setup

import pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pprint
import sys
sys.path.append('..')
import TDlib as td
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neural_network import MLPClassifier

accuracies_colors = [
    "xkcd:green",
    "xkcd:blue",
    "xkcd:brown",
    "xkcd:red"
]
accuracies_min_colors = [
    "xkcd:light green",
    "xkcd:periwinkle",
    "xkcd:light brown",
    "xkcd:light red"
]
accuracies_max_colors = [
    "xkcd:bright green",
    "xkcd:royal blue",
    "xkcd:mahogany",
    "xkcd:crimson"
]

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
X_np = data.loc[:, data.columns!=predict_column]
Y = data[predict_column]

#%%
# Describe data

# Nb individus
# Nb variables + type
# Nb individus par class

print(data.describe(percentiles=[]))
print(data.dropna().describe(percentiles=[]))
print("Écarts-type")
print(data.std())

td.renameCategories(data, categories).groupby('Classe').size().plot(kind='pie', autopct='%.2f %%')
plt.title("Répartition des classes")
plt.show()
td.renameCategories(data, categories).groupby('SousClasse').size().plot(kind='pie', autopct='%.2f %%')
plt.title("Répartition des sous-classes")
plt.show()

#%%
# Descriptive stats

#td.displayBoxplot(td.renameCategories(data, categories), "Classe", sharey=False)

# %%
# Separate train and test sets

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Re-diviser X_train en training et validation pour la phase de cross-validation
# Googler cross-validation PAS A LA MAIN
# Checker exemple dans cours
# Permet de tune kernel & param. C pour le SVM, ou alors le nb de neurones et la
# fct d'activation du MLP
# (tracer accuracies)

# Une fois qu'on a trouvé le meilleur SVM et le meilleur MLP, passer sur la base
# X_test pour les comparer

#%%
# ACP Classe

td.displayPopulationInFirstMainComponents(
    X.loc[:, X.columns!="SousClasse"],
    predict_column,
    ["Naturel", "Artificiel"]
)

# %%
# ALD Classe

td.displayPopulationInFirstAndRandomDiscriminantComponents(
    X.loc[:, X.columns!="SousClasse"],
    predict_column,
    ["Naturel", "Artificiel"]
)

# %%
# SVM

X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y)

kernels = ["linear", "poly", "rbf", "sigmoid"]

X_val = td.separateNumAndCat(X_train)['Xnum']

i = 0
for kernel in kernels :

    c_values = []
    accuracies = []
    accuracies_min_conf = []
    accuracies_max_conf = []

    for C in np.linspace(0.1, 10, 10):
        clf = svm.SVC(kernel=kernel, C=C, random_state=42)
        scores = cross_val_score(clf, X_val, Y_train, cv=5)

        print(f"{kernel}, C={C} : {scores.mean()*100:.2f} %")

        c_values.append(C)

        p = scores.mean()*100
        accuracies.append(p)

        conf_element = 1.96 * np.sqrt( p*(100-p) / len(Y_train) )

        accuracies_min_conf.append(p - conf_element)
        accuracies_max_conf.append(p + conf_element)

    plt.plot(
        c_values,
        accuracies,
        label=kernel,
        color=accuracies_colors[i]
    )
    plt.fill_between(
        c_values,
        accuracies_min_conf,
        accuracies_max_conf,
        color=accuracies_min_colors[i]
    )

    i += 1

plt.xlabel("C value")
plt.ylabel("Accuracy score (%)")
plt.title(f"Accuracy en fonction de C")
plt.legend()
plt.show()



#%%
# Testing SVM

C = 10
kernel = 'rbf'

X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y, test_size=0.5)

clf = svm.SVC(kernel=kernel, C=C)
clf.fit(X_train, Y_train)

true = Y_test
predict = clf.predict(X_test)

confmatrix_norm = confusion_matrix(
    true,
    predict,
    normalize='true'
)
tp, fp, fn, tn = confmatrix_norm.ravel()

disp = ConfusionMatrixDisplay(
    confmatrix_norm,
    display_labels=["NATUREL", "ARTIFICIEL"]
)
disp.plot()
plt.title(f"Matrice de confusion SVM C={C} kernel='{kernel}'")
plt.show()

print(f"TP = {tp}")
print(f"FP = {fp}")
print(f"TN = {tn}")
print(f"FN = {fn}")

# %%
# MLP

X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y)

activations = ["identity", "logistic", "tanh", "relu"]

X_val = td.separateNumAndCat(X_train)['Xnum']

i = 0
for activation in activations :

    hl_values = []
    accuracies = []
    accuracies_min_conf = []
    accuracies_max_conf = []

    for HL in range(1, 100, 10):
        mlp = MLPClassifier(
            hidden_layer_sizes=(HL,),
            activation=activation,
            max_iter=500
        )
        scores = cross_val_score(mlp, X_val, Y_train, cv=5)

        print(f"{activation}, HL={HL} : {scores.mean()*100:.2f} %")

        hl_values.append(HL)

        p = scores.mean()*100
        accuracies.append(p)

        conf_element = 1.96 * np.sqrt( p*(100-p) / len(Y_train) )

        accuracies_min_conf.append(p - conf_element)
        accuracies_max_conf.append(p + conf_element)

    plt.plot(
        hl_values,
        accuracies,
        label=activation,
        color=accuracies_colors[i]
    )
    plt.fill_between(
        hl_values,
        accuracies_min_conf,
        accuracies_max_conf,
        color=accuracies_min_colors[i]
    )

    i += 1

plt.xlabel("Hidden layer size")
plt.ylabel("Accuracy score (%)")
plt.title(f"Accuracy en fonction de la taille de la couche cachée")
plt.legend()
plt.show()


#%%
# Testing MLP

activation = "tanh"
HLS = 50

X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y)

mlp = MLPClassifier(activation=activation, hidden_layer_sizes=HLS)
mlp.fit(X_train, Y_train)

true = Y_test
predict = mlp.predict(X_test)

confmatrix_norm = confusion_matrix(
    true,
    predict,
    normalize='true'
)
tp, fp, fn, tn = confmatrix_norm.ravel()

disp = ConfusionMatrixDisplay(
    confmatrix_norm,
    display_labels=["NATUREL", "ARTIFICIEL"]
)
disp.plot()
plt.title(f"Matrice de confusion MLP HLS={HLS} activation='{activation}'")
plt.show()

print(f"TP = {tp}")
print(f"FP = {fp}")
print(f"TN = {tn}")
print(f"FN = {fn}")

#%%
# SVM SousClasse

predict_column = "SousClasse"
X_np = data.loc[:, data.columns!=predict_column]
Y = data[predict_column]

X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y)

kernels = ["linear", "poly", "rbf", "sigmoid"]

X_val = td.separateNumAndCat(X_train)['Xnum']

i = 0
for kernel in kernels :

    c_values = []
    accuracies = []
    accuracies_min_conf = []
    accuracies_max_conf = []

    for C in np.linspace(0.1, 10, 10):
        clf = svm.SVC(kernel=kernel, C=C, random_state=42)
        scores = cross_val_score(clf, X_val, Y_train, cv=5)

        print(f"{kernel}, C={C} : {scores.mean()*100:.2f} %")

        c_values.append(C)

        p = scores.mean()*100
        accuracies.append(p)

        conf_element = 1.96 * np.sqrt( p*(100-p) / len(Y_train) )

        accuracies_min_conf.append(p - conf_element)
        accuracies_max_conf.append(p + conf_element)

    plt.plot(
        c_values,
        accuracies,
        label=kernel,
        color=accuracies_colors[i]
    )
    plt.fill_between(
        c_values,
        accuracies_min_conf,
        accuracies_max_conf,
        color=accuracies_min_colors[i]
    )

    i += 1

plt.xlabel("C value")
plt.ylabel("Accuracy score (%)")
plt.title(f"Accuracy en fonction de C")
plt.legend()
plt.show()

#%%
# Testing SVM SousClasse

predict_column = "SousClasse"
X_np = data.loc[:, data.columns!=predict_column]
Y = data[predict_column]

C = 10
kernel = 'rbf'

X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y, test_size=0.5)

clf = svm.SVC(kernel=kernel, C=C)
clf.fit(X_train, Y_train)

true = Y_test
predict = clf.predict(X_test)

confmatrix_norm = confusion_matrix(
    true,
    predict,
    normalize='true'
)
# tp, fp, fn, tn = confmatrix_norm.ravel()

disp = ConfusionMatrixDisplay(
    confmatrix_norm,
    display_labels=["COTE", "FORET", "AUTOROUTE", "VILLE", "MONTAGNE", "OPEN_COUNTRY", "RUE", "GRANDBATIMENT"]
)
disp.plot()
plt.title(f"Matrice de confusion SVM C={C} kernel='{kernel}'")
plt.show()

# print(f"TP = {tp}")
# print(f"FP = {fp}")
# print(f"TN = {tn}")
# print(f"FN = {fn}")

# %%
# MLP SousClasse

predict_column = "SousClasse"
X_np = data.loc[:, data.columns!=predict_column]
Y = data[predict_column]

X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y)

activations = ["identity", "logistic", "tanh", "relu"]

X_val = td.separateNumAndCat(X_train)['Xnum']

i = 0
for activation in activations :

    hl_values = []
    accuracies = []
    accuracies_min_conf = []
    accuracies_max_conf = []

    for HL in range(1, 100, 10):
        mlp = MLPClassifier(
            hidden_layer_sizes=(HL,),
            activation=activation,
            max_iter=500
        )
        scores = cross_val_score(mlp, X_val, Y_train, cv=5)

        print(f"{activation}, HL={HL} : {scores.mean()*100:.2f} %")

        hl_values.append(HL)

        p = scores.mean()*100
        accuracies.append(p)

        conf_element = 1.96 * np.sqrt( p*(100-p) / len(Y_train) )

        accuracies_min_conf.append(p - conf_element)
        accuracies_max_conf.append(p + conf_element)

    plt.plot(
        hl_values,
        accuracies,
        label=activation,
        color=accuracies_colors[i]
    )
    plt.fill_between(
        hl_values,
        accuracies_min_conf,
        accuracies_max_conf,
        color=accuracies_min_colors[i]
    )

    i += 1

plt.xlabel("Hidden layer size")
plt.ylabel("Accuracy score (%)")
plt.title(f"Accuracy en fonction de la taille de la couche cachée")
plt.legend()
plt.show()

#%%
# Testing MLP Sous Classe

activation = "tanh"
HLS = 50

predict_column = "SousClasse"
X_np = data.loc[:, data.columns!=predict_column]
Y = data[predict_column]

X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y)

mlp = MLPClassifier(activation=activation, hidden_layer_sizes=HLS)
mlp.fit(X_train, Y_train)

true = Y_test
predict = mlp.predict(X_test)

confmatrix_norm = confusion_matrix(
    true,
    predict,
    normalize='true'
)
#tp, fp, fn, tn = confmatrix_norm.ravel()

disp = ConfusionMatrixDisplay(
    confmatrix_norm,
    display_labels=["COTE", "FORET", "AUTOROUTE", "VILLE", "MONTAGNE", "OPEN_COUNTRY", "RUE", "GRANDBATIMENT"]
)
disp.plot()
plt.title(f"Matrice de confusion MLP HLS={HLS} activation='{activation}'")
plt.show()

# print(f"TP = {tp}")
# print(f"FP = {fp}")
# print(f"TN = {tn}")
# print(f"FN = {fn}")

#%%
# PENSER AUX INTERVALLES DE CONFIANCE !!!!

# Validation : valider avec accuracy
# Testing : Checker la matrice de confusion complète
# Testing : Avec les paramètres définis en validation, refaire train_test_split
# sur la base complète

# Préciser le positif : Naturel, artificiel...

# Passer aux sous-classes avec les paramètres déterminés avec les classes
# Si bon : nickel
# Si pas bon : refaire une évaluation des paramètres SVM et MLP