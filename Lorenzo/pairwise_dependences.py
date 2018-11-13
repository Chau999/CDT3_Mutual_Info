import numpy as np
from sklearn.feature_selection import mutual_info_regression

def pairwise_corr(df):
    # each column of df represent a variable, each row an observation.
    
    num_vars = len(df.columns)
    
    return df.corr().as_matrix()[np.triu_indices(num_vars, k=1)]

def pairwise_kraskov_MI(df, n_neighbors=3):
    # each column of df represent a variable, each row an observation.
    # the returned vector contains -2 when all the elements of the considered pair are NaN, and it returns -1 when 
    # there are too few not NaN elements for the kNN algorithm to work.

    num_vars = len(df.columns)
    
    corr_matrix = np.zeros((num_vars, num_vars))
    
    for i in range(num_vars):
        for j in range(i+1, num_vars):
            pair = df.iloc[:,[i, j]].dropna()
            if len(pair) == 0:
                corr_matrix[i,j] = -2
            elif len(pair) <= n_neighbors: 
                corr_matrix[i,j] = -1
                #print(i,j, corr_matrix[i, j])
            else: 
                corr_matrix[i,j] = mutual_info_regression(pair.iloc[:,0].values.reshape(-1, 1), 
                                                          pair.iloc[:,1].values.reshape(-1, 1), n_neighbors=n_neighbors)
                #if (corr_matrix[i, j] == 0):
                    #print(i, j, len(pair), corr_matrix[i, j])
    return corr_matrix[np.triu_indices(num_vars, k=1)]