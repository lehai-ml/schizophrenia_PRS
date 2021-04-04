""" data_preprocessing.py
This is a custom files of all preprocessing steps necessary to transform the 
diffusion and volumetric data before the data exploration and ML training step.
"""

import os
import numpy as np
import pandas as pd

#import sklearn
from sklearn.linear_model import LinearRegression

def babies_connectivity_extraction(directory):
    """
    Extracting the babies ID_names and their diffusion data matrices
    
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

def read_table_from_txt(file,add_EP=False):
    
    """
    For reading PRS and ancestral PC tables with FID and IID columns in txt
    Args:
        file(.txt): path to the txt file
    
    Returns:
        table: table with FID removed, and ID sorted, matched with the babies 
            ID.
    
    """
    
    table=pd.read_table(file,delim_whitespace=True)
    if add_EP:
        table['IID']=['EP'+str(i) for i in table['IID']]
    table=table.drop('FID',axis=1)
    table=table.rename({'IID':'ID'},axis=1)
    table=table.sort_values('ID').reset_index(drop=True)
    return table

def move_multiple_columns(df,cols_to_move=[],ref_col='',place='After'):
    """
    Move multiple columns of the pandas dataframe to a new position.
    Args:
        df (Pandas dataframe)
        cols_to_move (list): list of the name of the columns
        ref_col (str): name of the reference column
        place ('After' or 'Before'): place the new columns either after or before the reference column.
    Returns:
        a new data frame.
    
    E.g.
        move_multiple_columns(european_diffusion_dataset,cols_to_move=['Session_y','Gender','GA','PMA'],ref_col='ID',place='After')
    """
    cols=df.columns.to_list()
    if place=='After':
        seg1=cols[:cols.index(ref_col)+1]
        seg2=cols_to_move
    if place=='Before':
        seg1=cols[:cols.index(ref_col)]
        seg2=cols_to_move+[ref_col]
    seg1=[i for i in seg1 if i not in seg2]
    seg3=[i for i in cols if i not in seg1+seg2]
    return(df[seg1+seg2+seg3])

def lower_triangle(matrix):
    """
    Organizes a square unidirectional matrix into 1D vector. Because the matrix 
        is unidirectional, only the lower triangle is needed.
    
    Args:
        matrix(np.array): the square matrix
    
    Returns:
        new_array(np.array): 1D vector of the lower half.
    Example:     
    0 1 2 3
    1 0 3 4
    2 3 0 5====>[1,2,3,3,4,5]
    3 4 5 0
    """
    m = matrix.shape[0]
    r = np.arange(m)
    mask = r[:,None] > r
    return matrix[mask]
    

def reverse_lower_triangle(matrix,side_of_the_square=90):
    """
    Organise a 1D matrix to a square 2D undirected matrix.
    Args:
        matrix: 1D matrix
        side_of_the_square (int): desired square size 
    Returns:
        matrix: 2D matrix
    """
    
    return_matrix=np.zeros((side_of_the_square,side_of_the_square))
    counter=0
    for i in range(1,side_of_the_square):
        for n in range(i):
            return_matrix[n][i]=return_matrix[i][n]=matrix[counter]
            counter+=1
    return return_matrix

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

def adjusting_for_covariates_with_lin_reg(y,*covariates):
    """
    Adjusting for covariates using linear regression.
    Args:
        y: Features to be adjusted
        covariates:covariates in np.array format
        
    Returns:
        new_feature: adjusted features
    """
    X=np.concatenate([i.reshape(-1,1) if i.ndim==1 else i for i in covariates],axis=1)
    lin_reg=LinearRegression()
    lin_reg.fit(X,y)
    y_pred=lin_reg.predict(X)
    new_feature=y-y_pred
    return new_feature

def common_features_across_all_splits(n,*args):
    """
    Find common features across all splits
    Args:
        n(int): number of repetition across n splits
        *args: the combination indices returned by the 5 splits. in 2D array.
    """
    unique_idx,unique_counts=np.unique(np.concatenate(args,axis=1),return_counts=True)
    return unique_idx[np.where(unique_counts>n)]

def high_low_risk_divide(y,high_perc=0.1,low_perc=0.3):
    
    """
    Divide the dataset into top and bottom x%. Here, we take the assumption that top percent is high risk and bottom percent is low risk.
    While this method maximises the odd ratio for impacts, it raises concerns about the arbitrariness of the quantile used.
    Args:
        y: the raw PRS score.
        high_perc (float): higher risk group percentage
        low_perc (float): lower risk group percentage
    Return:
        low_risk,high_risk (np.arrays): indices
    """
    
    high_risk_number=int(np.ceil(high_perc*len(y)))
    low_risk_number=int(np.ceil(low_perc*len(y)))
    if high_risk_number+low_risk_number>len(y):
        raise Exception('The high and low risk selection overlapped')
    #bottom
    low_risk=np.argsort(y)[:low_risk_number]
    #top
    high_risk=np.argsort(y)[::-1][:high_risk_number]
    
    return high_risk,low_risk