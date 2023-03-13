import inspect
import os.path as op
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def displayHistograms(
    X: pd.DataFrame,
    Xunits: dict,
    columns: list[str],
    title: str
):
    """Displays histograms side by side from columns of the DataFrame

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame from which to extract the data
    Xunits : dict
        List of units associated with the DataFrame column names
    columns : list[str]
        list of columns to display
    title : str
        Global title
    """
    Xnum = separateNumAndCat(X)['Xnum']

    Xnum[columns].hist()

    for i in range(0, len(columns)):
        ax = plt.subplot(1, len(columns), i+1)
        ax.set_xlabel("{} ({})".format(columns[i], Xunits[columns[i]]))
        ax.set_ylabel("Absolute value")

    plt.suptitle(title)

    plt.show()

def displayBarGraphs(
    X: pd.DataFrame,
    Xunits: dict,
    columns: list[str],
    title: str,
    show_zeros: bool = True,
):
    Xnum = separateNumAndCat(X)['Xnum']

    ###### NP.UNIQUE PERMET DE FAIRE APPARAITRES LES ZÉROS #########################

    plt.figure()

    for i in range(0, len(columns)):
        ax = plt.subplot(1, len(columns), i+1)
        if show_zeros:
            values, counts = np.unique(Xnum[columns[i]], return_counts=True)
            ax.bar(values, counts, width=0.5)
        else:
            Xnum[columns[i]].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel("{} ({})".format(columns[i], Xunits[columns[i]]))
        ax.set_ylabel("Absolute quantity")

    # ax2 = plt.subplot(1, 2, 2)
    # if show_zeros:
    #     values, counts = np.unique(Xnum[columns[1]], return_counts=True)
    #     ax2.bar(values, counts, width=0.5)
    # else:
    #     Xnum[columns[1]].value_counts().plot(kind='bar', ax=ax2)
    # ax2.set_xlabel(xlabels[1])
    # ax2.set_ylabel("Absolute quantity")

    plt.suptitle(title)

    plt.show()

def renameCategories(X: pd.DataFrame, categories: dict[dict]) -> pd.DataFrame:
    """Renames categories based on a dict

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame in which to rename the categories. They MUST be of dtype
        "category" to be affected
    categories : dict[dict]
        A dictionary with the desired DataFrame's column names as keys with
        another dict as value, which itself contains the categories to replace
        as keys and the new names as values

    Returns
    -------
    pd.DataFrame
        The new DataFrame with renamed categories
    """
    temp = X
    for column in temp.columns:
        temp[column] = temp[column].cat.rename_categories(categories[column])

    return temp

def displayTwoColumnScatter(
    X: pd.DataFrame,
    Xunits: dict,
    columns: tuple[str],
    title: str
):
    """Displays a scatter plot of two columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which the columns will be used
    Xunits: dict
        Units associated with the DataFrame columns
    columns: tuple[str]
        The X and Y column names
    title: str
        The graph's title
    """

    Xnum = separateNumAndCat(X)['Xnum']

    Xnum.plot.scatter(x=columns[0], y=columns[1])

    plt.xlabel("{} ({})".format(columns[0], Xunits[columns[0]]))
    plt.ylabel("{} ({})".format(columns[1], Xunits[columns[1]]))

    plt.title(title)

    plt.show()

def displayScatterMatrix(X: pd.DataFrame, title: str):
    """Displays a scatter matrix across all of the DataFrame's columns

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to extract the data from
    title: str
        The title of the graph
    """

    Xnum = separateNumAndCat(X)['Xnum']

    pd.plotting.scatter_matrix(Xnum)

    plt.suptitle(title)

    plt.show()

def displayCorrelationMatrix(X: pd.DataFrame, title: str):
    """Displays the correlation matrix of the numerical data in the DataFrame

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame from which to extract the data
    title : str
            The title of the graph
    """

    Xnum = separateNumAndCat(X)['Xnum']

    sns.heatmap(Xnum.corr(), cmap='coolwarm', annot=True)
    plt.title(title)
    plt.show()