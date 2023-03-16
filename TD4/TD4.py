#!/usr/bin/python

#%%
# Import modules
import sys
import os.path as op
libdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(libdir)
import TDlib as td
from pprint import pprint

# %%
# Import data

X = td.excelToPandas(
    "INFARCTU.xlsx",
    "INFARCTU",
    0,
    0,
    ['C', 'PRONO']
)

#%%
# 1/ Présenter le boxplot de chaque variable en tenant compte de la variable Prono

td.displayBoxplot(X, 'PRONO', sharey=False)

# %%
