import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def pp():
    list_csv = glob.glob('csv/*.csv')
    dfs = []

    for i in (list_csv):
        df = pd.read_csv(i)
        dfs.append(df)

    column_names = dfs[0].columns.tolist()

    return dfs, column_names