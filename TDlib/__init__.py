import inspect
import os.path as op
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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

def printMeanAndVar(X: pd.DataFrame, groupby: str = None):
    Xnum = separateNumAndCat(X)['Xnum']

    mean, var = 0, 0
    string = ""

    if groupby == None:
        mean = Xnum.mean()
        var  = Xnum.var()
    else:
        Xnum[groupby] = X[groupby].astype("category")

        mean = Xnum.groupby(groupby).mean()
        var  = Xnum.groupby(groupby).var()

        string = ", grouped by {}".format(groupby)

    printprefix("Means of the numerical variables{} :".format(string))
    print(mean.to_string())
    printprefix("Variances of the numerical variables{} :".format(string))
    print(var.to_string())

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
    newX = X.copy(deep=True)
    for column in newX.columns:
        if newX.dtypes[column] == 'category':
            newX[column] = newX[column].cat.rename_categories(categories[column])

    return newX

def displayTwoColumnScatter(
    X: pd.DataFrame,
    Xunits: dict,
    columns: tuple[str],
    title: str
):
    """Displays a scatter plot of two columns in the DataFrame.

^    Parameters
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

def displayScatterMatrix(X: pd.DataFrame, title: str, column: str=None):
    """Displays a scatter matrix across all of the DataFrame's columns

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to extract the data from
    title: str
        The title of the graph
    """

    Xnum = separateNumAndCat(X)['Xnum']

    if column != None:
        Xnum[column] = X[column].astype('category')
        pd.plotting.scatter_matrix(Xnum, c=Xnum[column])
    else:
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

def displayCovarianceMatrix(X: pd.DataFrame, title: str):
    """Displays the covariance matrix of the numerical data in the DataFrame

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to work on
    title : str
        The title of the graph
    """

    Xnum = separateNumAndCat(X)['Xnum']

    sns.heatmap(Xnum.cov(), cmap='coolwarm', annot=True)
    plt.title(title)
    plt.show()

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

def calculateGroupedMeans(
    X: pd.DataFrame,
    categories: dict[dict],
    groupby: str
) -> pd.DataFrame:
    """Returns a DataFrame containing groups as rows and means as columns

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to work on
    categories: dict[dict]
        See renameCategories
    grouby: str
        The column to group by

    Returns
    -------
    pd.DataFrame
        The grouped means
    """

    return renameCategories(X, categories).groupby(groupby).mean(numeric_only=True)

def displayGroupedBoxplot(
    X: pd.DataFrame,
    categories: dict[dict],
    Xunits: dict,
    groupby: str,
    columns: list[str]
):

    renameCategories(X, categories).boxplot(column=columns, by=groupby)
    plt.show()

def displayParetoDiagram(X: pd.DataFrame, title: str, center: bool = False):
    """Display Pareto Diagram for a DataFrame

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to work on
    title: str
        The title of the graph
    center: bool
        Wether to center and reduce the data beforehand or not
    """

    acp = PCA()

    if center:
        Xcr = scale(X, with_mean=True, with_std=True)
    else:
        Xcr = X

    Xacp = acp.fit(Xcr).transform(Xcr)

    y = list(acp.explained_variance_ratio_)
    x = range(len(y))
    ycum = np.cumsum(y)

    plt.bar(x, y)
    plt.plot(x, ycum, "-r")

    plt.xlabel("Number of factors (or eigenvectors)")
    plt.ylabel("Explained variances and cumulative explained variance")
    plt.title(title)

    plt.show()

def displayPopulationInFirstMainComponents(
    X: pd.DataFrame,
    column: str = "",
    labels: list[str] = [],
    center: bool = False
):
    """Displays data in the plane of the first two main PCA components

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to work on
    """

    acp = PCA()

    if center:
        Xcr = scale(X, with_mean=True, with_std=True)
    else:
        Xcr = X

    Xacp = acp.fit(Xcr).transform(Xcr)

    if column == "":
        plt.scatter(Xacp[:, 0], Xacp[:, 1])
    else:
        plt.scatter(
            Xacp[:, 0],
            Xacp[:, 1],
            c=X[column],
            label=labels
        )

#    for i, label in enumerate(X.index):
#        plt.annotate(label, (Xacp[i,0], Xacp[i,1]))

    plt.xlabel("Main component 1")
    plt.ylabel("Main component 2")
    plt.title("Population in the plane of first two main components")
    plt.show()

def displayCorrelationCircle(X: pd.DataFrame, center: bool = False):
    # corvar est de dimension (n,2) : contient dans la colonne 0 : la corrélation entre la composante principale 1 et les variables de départ 
    # et dans la colonne 1 la corrélation entre la composante principale 2 et les variables de départ

    acp = PCA()

    if center:
        Xcr = scale(X, with_mean=True, with_std=True)
    else:
        Xcr = X

    Xacp = acp.fit(Xcr).transform(Xcr)

    p = X.shape[1]

    corvar = np.zeros((p,2))

    for i in range(p):
        for j in range(2):
            corvar[i,j] = np.corrcoef(X.iloc[:,i], Xacp[:,j])[0,1]

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
    plt.xlabel("Main component 1")
    plt.ylabel("Main component 2")
    plt.title('Correlation circle')
    plt.scatter(corvar[:,0],corvar[:,1])
    #affichage des étiquettes (noms des variables)
    for j in range(p):
        plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

    plt.show()

def totalVariance(X: pd.DataFrame, center: bool = False) -> float:

    acp = PCA()

    if center:
        Xcr = scale(X, with_mean=True, with_std=True)
    else:
        Xcr = X

    Xacp = acp.fit(Xcr).transform(Xcr)

    return acp.explained_variance_.sum()    # First option
    # return X.var().sum()                  # Second option, NOT GOOD IF
    # CENTERED !!

def displayPopulationInFirstDiscriminantComponents(
    X: pd.DataFrame,
    column: str,
    labels: list[str],
    center: bool = False
):

    lda = LinearDiscriminantAnalysis()
    coord_lda = lda.fit_transform(X, X[column])

    plt.scatter(
        coord_lda[:,0], coord_lda[:,1],
        c=X[column], label=labels
    )
    plt.legend()
    plt.xlabel("Discriminant Component 1")
    plt.ylabel("Discriminant Component 2")
    plt.title("Population in the plane of first two discriminant components")
    plt.show()

def displayPopulationInFirstAndRandomDiscriminantComponents(
    X: pd.DataFrame,
    column: str,
    labels: list[str],
    center: bool = False
):

    lda = LinearDiscriminantAnalysis()
    coord_lda = lda.fit_transform(X, X[column])

    rand = np.random.randn(int(np.shape(coord_lda)[0]))

    plt.scatter(
        coord_lda[:,0], rand,
        c=X[column], label=labels
    )
    plt.legend()
    plt.xlabel("Discriminant Component 1")
    plt.ylabel("Random Component")
    plt.title("Population in the plane of first discriminant component and random value")
    plt.show()

def displayLDACorrelationCircle(X: pd.DataFrame, column: str):
    # corvar est de dimension (n,2) : contient dans la colonne 0 : la corrélation entre la composante principale 1 et les variables de départ 
    # et dans la colonne 1 la corrélation entre la composante principale 2 et les variables de départ

    p = X.shape[1]

    lda = LinearDiscriminantAnalysis()
    coord_lda = lda.fit_transform(X, X[column])

    corvar = np.zeros((p,2))

    for i in range(p):
        for j in range(2):
            corvar[i,j] = np.corrcoef(X.iloc[:,i], coord_lda[:,j])[0,1]

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
    plt.xlabel("Composante discriminante 1")
    plt.ylabel("Composante discriminante 2")
    plt.title('Cercle des corrélations')
    plt.scatter(corvar[:,0],corvar[:,1])
    #affichage des étiquettes (noms des variables)
    for j in range(p):
        plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

    plt.show()

def displayConfusionMatrix(X: pd.DataFrame, column: str):

    lda = LinearDiscriminantAnalysis()
    coord_lda = lda.fit_transform(X.loc[:, X.columns!=column], X[column])

    true = X[column]
    predict = lda.predict(X.loc[:, X.columns!=column])

    confmatrix_norm = confusion_matrix(true, predict, normalize='true')

    disp = ConfusionMatrixDisplay(confmatrix_norm)
    disp.plot()

def specificity_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return tn / (tn+fp)