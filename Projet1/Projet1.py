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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

#%%
# DATA FORMATTING

# data est un dictionnaire contenant 3 jeux de données data1, data2, data3
# data1 est un tableau contenant un tableau par forme d'onde
# Chacun de ces tableaux contient 9 paramètres extraits de la forme d'onde

data = scipy.io.loadmat('data.mat')

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
###### CLASSIFIER 1 : LDA

column = 'Valid'

lda = LinearDiscriminantAnalysis()
coord_lda = lda.fit_transform(X1.loc[:, X1.columns!=column], X1[column])

true = X1[column]
predict = lda.predict(X1.loc[:, X1.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

print("On observe que la prédiction LDA est capable de prédire de manière assez fiable si un pouls est correct. Pour les artefact, c'est moins évident.")

true = X2[column]
predict = lda.predict(X2.loc[:, X2.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

true = X3[column]
predict = lda.predict(X3.loc[:, X3.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

print("Intra-sujet, prediction OK, mais inter-sujet : pas ouf")

tp_lda = (true, predict)

#%%
###### CLASSIFIER 2 : QDA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
coord_qda = qda.fit(X1.loc[:, X1.columns!=column], X1[column])

true = X1[column]
predict = qda.predict(X1.loc[:, X1.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

true = X2[column]
predict = qda.predict(X2.loc[:, X2.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

true = X3[column]
predict = qda.predict(X3.loc[:, X3.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

tp_qda = (true, predict)

#%%
###### CLASSIFIER 3 : GNB

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
coord_gnb = gnb.fit(X1.loc[:, X1.columns!=column], X1[column])

true = X1[column]
predict = gnb.predict(X1.loc[:, X1.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

true = X2[column]
predict = gnb.predict(X2.loc[:, X2.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

true = X3[column]
predict = gnb.predict(X3.loc[:, X3.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

tp_gnb = (true, predict)

#%%
###### CLASSIFIER 4 : KNC

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
coord_gnb = knc.fit(X1.loc[:, X1.columns!=column], X1[column])

true = X1[column]
predict = knc.predict(X1.loc[:, X1.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

true = X2[column]
predict = knc.predict(X2.loc[:, X2.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

true = X3[column]
predict = knc.predict(X3.loc[:, X3.columns!=column])

confmatrix_norm = confusion_matrix(true, predict, normalize='true')

disp = ConfusionMatrixDisplay(confmatrix_norm)
disp.plot()

tp_knc = (true, predict)

#%%
###### SUMMARY

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from prettytable import PrettyTable
from prettytable import DOUBLE_BORDER
table = PrettyTable()
table.set_style(DOUBLE_BORDER)

table.field_names = ["Confusion Matrix Score", "LDA", "QDA", "GNB", "KNC"]
table.add_row([
    "Accuracy",
    round(accuracy_score(tp_lda[0], tp_lda[1]), 2),
    round(accuracy_score(tp_qda[0], tp_qda[1]), 2),
    round(accuracy_score(tp_gnb[0], tp_gnb[1]), 2),
    round(accuracy_score(tp_knc[0], tp_knc[1]), 2)
])
table.add_row([
    "Precision",
    round(precision_score(tp_lda[0], tp_lda[1]), 2),
    round(precision_score(tp_qda[0], tp_qda[1]), 2),
    round(precision_score(tp_gnb[0], tp_gnb[1]), 2),
    round(precision_score(tp_knc[0], tp_knc[1]), 2)
])
table.add_row([
    "Recall",
    round(recall_score(tp_lda[0], tp_lda[1]), 2),
    round(recall_score(tp_qda[0], tp_qda[1]), 2),
    round(recall_score(tp_gnb[0], tp_gnb[1]), 2),
    round(recall_score(tp_knc[0], tp_knc[1]), 2)
])
table.add_row([
    "Specificity",
    round(td.specificity_score(tp_lda[0], tp_lda[1]), 2),
    round(td.specificity_score(tp_qda[0], tp_qda[1]), 2),
    round(td.specificity_score(tp_gnb[0], tp_gnb[1]), 2),
    round(td.specificity_score(tp_knc[0], tp_knc[1]), 2)
])
table.add_row([
    "F1",
    round(f1_score(tp_lda[0], tp_lda[1]), 2),
    round(f1_score(tp_qda[0], tp_qda[1]), 2),
    round(f1_score(tp_gnb[0], tp_gnb[1]), 2),
    round(f1_score(tp_knc[0], tp_knc[1]), 2)
])

print(table)


#%%

column = 'Valid'

######### LDA
lda = LinearDiscriminantAnalysis()
coord_lda = lda.fit_transform(X1.loc[:, X1.columns!=column], X1[column])

true = X3[column]
predict = lda.predict_proba(X3.loc[:, X3.columns!=column])[:, 1]

from sklearn.metrics import roc_curve

plt.figure()

FP, TP, TH = roc_curve(true, predict, pos_label=2)
plt.plot(FP, TP, label="LDA")

######### QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
coord_qda = qda.fit(X1.loc[:, X1.columns!=column], X1[column])

true = X3[column]
predict = qda.predict_proba(X3.loc[:, X3.columns!=column])[:, 1]

FP, TP, TH = roc_curve(true, predict, pos_label=2)
plt.plot(FP, TP, label="QDA")

######### GNB
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
coord_gnb = gnb.fit(X1.loc[:, X1.columns!=column], X1[column])

true = X3[column]
predict = gnb.predict_proba(X3.loc[:, X3.columns!=column])[:, 1]

FP, TP, TH = roc_curve(true, predict, pos_label=2)
plt.plot(FP, TP, label="GNB")


######### KNC
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
coord_gnb = knc.fit(X1.loc[:, X1.columns!=column], X1[column])

true = X3[column]
predict = knc.predict_proba(X3.loc[:, X3.columns!=column])[:, 1]

FP, TP, TH = roc_curve(true, predict, pos_label=2)
plt.plot(FP, TP, label="KNC")

######### Random
x_values = np.linspace(0, 1, X1.shape[0])
plt.plot(x_values, x_values, label="Random classifier")

######### Ideal
ideal_classifier = np.ones(X1.shape[0])
ideal_classifier[0] = 0
plt.plot(x_values, ideal_classifier, label="Ideal classifier")

plt.xlabel("Probabilité de fausse alarme")
plt.ylabel("Probabilité de bonne détection")
plt.legend()
plt.show()


# %%

print(X1)

#%%
print(X2)

#%%
print(X3)

# %%
#%%
# NOTEZ

# Courbes ROC !!!