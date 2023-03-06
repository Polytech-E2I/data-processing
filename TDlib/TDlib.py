import inspect
import os.path as op
import pandas as pd
from pprint import pprint

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