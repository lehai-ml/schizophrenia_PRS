"""
This is a custom files of all preprocessing steps prior to the ML training.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

#import sklearn
from sklearn.linear_model import LinearRegression

def babies_connectivity_extraction(directory):
    """
    Extracting the babies ID_names and connectivity matrices
    
    Args:
        directory(str): path to the tracts information
    
    Returns:
        ID (list): the name of the babies
        connectivity_matrix(list): list of connectivity matrices (90 x90)
    """
    ID=[]
    connectivity_matrix=[]
    for file in os.listdir(directory):
        try:
            input_ = np.loadtxt(directory+file, dtype=np.float64, delimiter=' ')
            id_=str.split(file,'_')[1]
            if 'dataProb' in file: # make sure to remove these files
                print('this file %s is not processed'%file)
                continue
            if id_ in ID:
                print ('this id %s is repeated'%id_)
            else:
                connectivity_matrix.append(input_)
                ID.append(id_)
            if input_.shape[0]!=90|(input_.shape[1])!=90:
                print('this file doesnt have the 90x90 shape:%s'%file)
        except UnicodeDecodeError:
            print('this file cannot be loaded: %s'%file)
    return ID, connectivity_matrix

def ROIs_combinations(csv_filename):
    """
    Provide the combination of the Region of Interests.
    
    Args:
        csv_filename(str): path to the filename in .csv
    
    Returns:
    
        combinations (list): list of combination in the same order as the 
            connectivity matrices (90x90)
    """
    # import regions of interest names
    ROIs=pd.read_csv(csv_filename).dropna()
    ROIs=ROIs.loc[:,'abbr. '].values
    ROIs=[str.split(i,' ')[0] for i in ROIs]
    ROIs=['.'.join(str.split(i,'.')) for i in ROIs]
    combinations=[]
    for i in ROIs:
        combination=[i+'_'+n for n in ROIs]
        combinations.append(combination)
    combinations=np.vstack(combinations)
    return combinations

def read_table_from_txt(file):
    
    """
    For reading tables with FID and IID columns in txt
    Args:
        file(.txt): path to the txt file
    
    Returns:
        table: table with FID removed, and ID sorted, matched with the babies 
            ID.
    
    """
    
    table=pd.read_table(file,delim_whitespace=True)
    table['IID']=['EP'+str(i) for i in table['IID']]
    table=table.drop('FID',axis=1)
    table=table.rename({'IID':'ID'},axis=1)
    table=table.sort_values('ID').reset_index(drop=True)
    return table

def lower_triangle(matrix,side_of_the_square=90):
    """
    Organizes a square unidirectional matrix into 1D vector. Because the matrix 
        is unidirectional, only the lower triangle is needed.
    
    Args:
        matrix(np.array): the square matrix
        side_of_the_square(int): the side of the square. 90 (Default)
    
    Returns:
        new_array(np.array): 1D vector of the lower half.
    """
    lower_triangle=[matrix[i][n]
                  for i in range(1,side_of_the_square) for n in range(i)]
    return np.asarray(lower_triangle)

def extract_by_ID(ID_list,original_ID_array,original_array,remove=False):
    """
    Extract by the babies ID, required for matching by ID and after outliers
    
    Args:
        ID_list(list): the list of ID that need to be matched to
        original_ID_array(list): the list of ID that needs to be modified
        original_array(list): the array that needs to be modified
        remove(bool): Do you want to remove those ID?
    
    Returns:

        new_array
    """
    indices=[np.where(np.asarray(original_ID_array)==i)[0][0] for i in ID_list]
    if remove:
        new_array=np.delete(np.asarray(original_array),indices)
    else:
        new_array=np.asarray(original_array)[indices]
    return new_array

def adjusting_for_covariates_with_lin_reg(y,covariates=([])):
    """
    Adjusting for covariates using linear regression.
    Args:
        y: Features to be adjusted
        covariates(list):list of covariates
        
    Returns:
        new_feature: adjusted features
    """
    X=np.concatenate([i.reshape(-1,1) if i.ndim==1 else i for i in covariates],axis=1)
    lin_reg=LinearRegression()
    lin_reg.fit(X,y)
    y_pred=lin_reg.predict(X)
    new_feature=y-y_pred
    return new_feature

def common_features_across_all_splits(n=4,axis=1,*args):
    """
    Find common features across all splits
    Args:
        n(int): number of repetition across 5 splits
        *args: the combination indices returned by the 5 splits. in 2D array.
    """
    unique_idx,unique_counts=np.unique(np.concatenate(args,axis=axis),return_counts=True)
    return unique_idx[np.where(unique_counts>n)]

def high_low_risk_divide(y,bins=10,percentage=0.2):
    """
    Divide the dataset into top and bottom x%. Here, we take the assumption that top percent is high risk and bottom percent is low risk.
    Args:
        y: the raw PRS score.
        bins: number of bins to divide.
        percentage (float)
    Return:
        low_risk,high_risk (np.arrays): indices
        
    """
    
    counts,bin=np.histogram(y,bins=bins)
    number_to_retain=percentage*len(y)
    #bottom
    pos_low=np.where(np.cumsum(counts)>=number_to_retain)[0][0]
    low_risk=np.where(y<=bin[pos_low+1])[0]
    #top
    pos_high=np.where(np.cumsum(counts[::-1])>=number_to_retain)[0][0]
    high_risk=np.where(y>=bin[::-1][pos_high+1])[0]
    
    return low_risk,high_risk 



if __name__ == "__main__":
    pass
    