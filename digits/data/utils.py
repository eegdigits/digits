import numpy as np
import pandas as pd

def remove_duplicate_columns(df, factor=2):

    row1 = df.iloc[0:]
    column_pairs = df.columns.values
    unique_pairs = np.unique(df.columns.values)
    if len(column_pairs) == len(unique_pairs):
        print('column/level names are already unique')
        return df

    suffixes = ['.'+str(a) for a in np.arange(1,factor+1)]
    newnames = [ (x, y+suffixes[ix%factor]) for (ix,(x,y)) in enumerate(list(df.columns)) ]
    newindex = pd.MultiIndex.from_tuples(tuples = newnames, names=df.columns.names)
    df.columns = newindex
    df.columns.set_labels = np.tile(np.arange(2*len(newnames)), len(row1.columns.levels[0]))
    return df
