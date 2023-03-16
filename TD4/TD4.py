#!/usr/bin/python

#%%
# Import modules
import sys
import os.path as op
libdir = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(libdir)
import TDlib as td
import pandas as pd

# %%
# Import data

X = td.excelToPandas(
    "INFARCTU.xlsx",
    "INFARCTU",
    0,
    0,
    ['C', 'PRONO']
)