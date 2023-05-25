import matplotlib.pyplot as plt
import pandas as pd


def separateNumAndCat(X: pd.DataFrame) -> dict[pd.DataFrame]:
    return {
        'Xnum': X.select_dtypes(exclude=['category']),
        'Xcat': X.select_dtypes(include=['category'])
    }


def displayBoxplot(X: pd.DataFrame, groupby: str = None, sharey: bool = True):
    Xnum = separateNumAndCat(X)['Xnum']

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