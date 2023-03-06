import inspect
import os.path as op
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

def printprefix(message: str):
    PREFIX="  ### "

    print(PREFIX + message)

def excelToPandas(
    file: str,
    sheet: str,
    header: int,
    index_col: int,
    categories: list[str]
) -> pd.DataFrame:
    # Get the caller script's path
    callerfilename = op.abspath(inspect.stack()[1].filename)
    callerdir = op.dirname(callerfilename)

    X = pd.read_excel(
        op.join(callerdir, file),
        sheet_name=sheet,
        header=header,
        index_col=index_col
    )
    X.dropna()

    X[categories] = X[categories].astype("category")

    return X

def separateNumAndCat(X: pd.DataFrame) -> dict[pd.DataFrame]:
    return {
        'Xnum': X.select_dtypes(exclude=['category']),
        'Xcat': X.select_dtypes(include=['category'])
    }

def prettyPrintDataframe(X: pd.DataFrame):
    Xcat = separateNumAndCat(X)['Xcat']
    Xnum = separateNumAndCat(X)['Xnum']

    printprefix("Numerical variables :")
    pprint(Xnum)
    printprefix("Category variables :")
    pprint(Xcat)

def printMeanAndVar(X: pd.DataFrame):
    Xnum = separateNumAndCat(X)['Xnum']

    printprefix("Means of the numerical variables :")
    print(Xnum.mean().to_string())
    printprefix("Variances of the numerical variables :")
    print(Xnum.var().to_string())

def displayBoxplot(X: pd.DataFrame):
    Xnum = separateNumAndCat(X)['Xnum']

    Xnum.boxplot()
    plt.ylabel("Values")
    plt.title("Spread of numerical variables")

def displayDualHistograms(
    X: pd.DataFrame,
    columns: list[str],
    xlabels: list[str],
    title: str
):
    Xnum = separateNumAndCat(X)['Xnum']

    Xnum[columns].hist()

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlabel(xlabels[0])
    ax1.set_ylabel("Absolute quantity")
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_xlabel(xlabels[1])
    ax2.set_ylabel("Absolute quantity")
    plt.suptitle(title)

    plt.show()

def displayDualBarGraph(
    X: pd.DataFrame,
    columns: list[str],
    xlabels: list[str],
    title: str,
    show_zeros: bool = False,
):
    Xnum = separateNumAndCat(X)['Xnum']

    ###### NP.UNIQUE PERMET DE FAIRE APPARAITRES LES ZÉROS #########################

    plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    if show_zeros:
        #values, counts = np.unique(Xnum.iloc[:, 2], return_counts=True)
        values, counts = np.unique(Xnum[columns[0]], return_counts=True)
        ax1.bar(values, counts, width=0.5)
    else:
        Xnum[columns[0]].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_xlabel(xlabels[0])
    ax1.set_ylabel("Absolute quantity")

    ax2 = plt.subplot(1, 2, 2)
    if show_zeros:
        values, counts = np.unique(Xnum[columns[1]], return_counts=True)
        ax2.bar(values, counts, width=0.5)
    else:
        Xnum[columns[1]].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_xlabel(xlabels[1])
    ax2.set_ylabel("Absolute quantity")

    plt.suptitle(title)

    plt.show()

def displayScatterMatrix(
    X: pd.DataFrame,
    columns: list[str]
):
    Xnum = separateNumAndCat(X)['Xnum']

    Xnum.plot.scatter(x=columns[0], y=columns[1])
    pd.plotting.scatter_matrix(Xnum)