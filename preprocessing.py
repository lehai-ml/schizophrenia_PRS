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


class ROIs_combo:
    """
    All processing of the ROIs combinations
    Built using networkx library
    """
    def __init__(self, combinationsarray,connectivityarray=None):
        """
        Returns:
        self.ROIs_combination
        self.unique_regions= return regions names
        self.unique_regions_count= return regions count
        self.unique_selected_brain_structures=return brain structure name
        self.unique_selected_brain_structures_total_counts=return brain structure count
        self.Graph=returns networkx object graph.
        """
        self.ROIs_combination=combinationsarray
        if all(connectivityarray)!=None:
            non_zero_idx=np.where(connectivityarray!=0)
            self.ROIs_combination=combinationsarray[non_zero_idx]
            self.connectivityarray=connectivityarray[non_zero_idx]
        else:
            self.connectivityarray=np.ones(self.ROIs_combination.shape)#if the connectivity vector is not available, have a placeholder as np.ones
            
        self.unique_regions,self.unique_regions_count,self.unique_selected_brain_structures,self.unique_selected_brain_structures_total_counts=self.extract_individual_regions_from_connections()
        self.Graph=self.create_network()
        
    def extract_individual_regions_from_connections(self):
        """
        Given a list of ROIs combinations, return the unique brain structure names
        Input: combinationsarray of the connections combinations
        Output: Return unique_regions, unique_regions_counts, unique_brain_structures, and unique_brain_structures_counts.
        """
        single_regions=np.concatenate((np.asarray([str.split(i,'_') for i in self.ROIs_combination])),axis=0) #for each of the combinations, divide them to get the individual regions, then concatenate them into a big list.
        unique_regions,unique_regions_count=np.unique(single_regions,return_counts=True) # get the unique region names, and return the counts.

        selected_brain_structures_temps=np.asarray([str.split(i,'.')[0] for i in unique_regions])#the '.' separate the brain name from hemisphere sid, so here I extract only the brain name from the unique regions list
        unique_selected_brain_structures=np.unique(selected_brain_structures_temps,return_counts=True,return_index=True)# from the list of brain names, I return the counts and indices, in this way, ACG.R and ACG.L will be counted as 1 region ACG.
        unique_selected_brain_structures_total_counts=np.asarray([np.sum(unique_regions_count[range(idx,idx+counts)]) for idx, counts in (zip(unique_selected_brain_structures[1],unique_selected_brain_structures[2]))])#since the unique_regions will be sorted,i.e. ACG.L will be followed by ACG.R, I can get the number of connections from/to ACG by summing the counts of those 2 regions and using their index.

        return unique_regions,unique_regions_count,unique_selected_brain_structures[0],unique_selected_brain_structures_total_counts
            
    def create_network(self):
        """
        Built using networkx library

        INPUT: requires, ROIs_combinations to create edge list
        connectivityarray to define the width of the edges
        Output: networkx object graph G.
        Use networkx.info(G) to get more information 
        
        """
        Graph=nx.MultiGraph()#this is undirected graph
        source_list=[]
        target_list=[]
        weight_list=self.connectivityarray
        edge_list=[str.split(i,'_') for i in self.ROIs_combination]#this is the list of edges
        for edge in edge_list:
            source_list.append(edge[0])
            target_list.append(edge[1])
        edges_df=pd.DataFrame(
            {
                "source":source_list,
                "target":target_list,
                "weight":weight_list
            }
        )
        Graph=nx.from_pandas_edgelist(edges_df,edge_attr=True)
        
        return Graph
    




if __name__ == "__main__":
    pass
    