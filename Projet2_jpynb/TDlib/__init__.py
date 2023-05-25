import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def separateNumAndCat(X: pd.DataFrame) -> dict[pd.DataFrame]:
    return {
        'Xnum': X.select_dtypes(exclude=['category']),
        'Xcat': X.select_dtypes(include=['category'])
    }


def displayBoxplot(X: pd.DataFrame, groupby: str = None, sharey: bool = True):
    #Xnum = separateNumAndCat(X)['Xnum']
    Xnum=X

    if groupby == None:
        Xnum.boxplot()
    else:
        Xnum[groupby] = X[groupby].astype("category")

        Xnum.boxplot(by=groupby, sharey=sharey)

    plt.ylabel("Values")
    plt.title("Spread of numerical variables")

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
    newX = X.copy(deep=True)
    for column in newX.columns:
        if newX.dtypes[column] == 'category':
            newX[column] = newX[column].cat.rename_categories(categories[column])

    return newX

def displayCrosstab(
    X: pd.DataFrame,
    categories: dict[dict],
    columns: tuple[str],
    title: str
):
    """Display cross tabulation of two columns

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame from which to extract the data
    categories : dict[dict]
        A dictionary with the desired DataFrame's column names as keys with
        another dict as value, which itself contains the categories to replace
        as keys and the new names as values
    columns : tuple[str]
        Tuple of the desired column names
    title : str
        Title of the graph
    """

    X_renamed = renameCategories(X, categories)

    crosstab = pd.crosstab(X_renamed[columns[0]], X_renamed[columns[1]])

    crosstab_prop = pd.crosstab(
        X_renamed[columns[0]],
        X_renamed[columns[1]],
        normalize="index"
    )

    crosstab_prop.plot(kind="bar", stacked=True, legend=True)

    # Display quantities and proportions over bars
    for n, x in enumerate([*crosstab.index.values]):
        for (proportion, count, y_loc) in zip(
                                              crosstab_prop.loc[x],
                                              crosstab.loc[x],
                                              crosstab_prop.loc[x].cumsum()):

            plt.text(
                x=n - 0.10,
                y=(y_loc - proportion) + (proportion / 2),
                s="{}\n({}%)".format(count, np.round(proportion*100, 1)),
                color="black",
                fontsize=12,
                fontweight="bold"
            )

    plt.ylabel("Proportion")
    plt.title(title)
    plt.show()
