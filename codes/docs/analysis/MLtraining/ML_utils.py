"""ML_utils.py
This custom file contains functions to run scikit-learn preprocessing pipelines 
    and training ML models.
"""
import pandas as pd
import numpy as np

#Scikit-lib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold,SelectorMixin

#Network visualisation and algorithm
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

#Python essentials

import operator
from itertools import combinations

#Custom functions
from .. import data_preprocessing

def lowest_percent_variance(percent,variance_object):
    
    """
    ___________________________________________________________
    Returns a VarianceThreshold transfomer with a new threshold.
    ___________________________________________________________
    Args:
        percent (float): The percentage of variances you want to remove.
        variance_object(VarianceThreshold()): The fitted to data
            VarianceThreshold()
    
    Returns:
        variance_object: A new VarianceThreshold() with a new defined threshold.
    """
    variance_list=np.unique(variance_object.variances_) #this will give a sorted list of all feature variances
    new_threshold_idx=int(np.ceil(len(variance_list)*percent)) # this will give me the index of the new threshold (so if there are 100 unique variances and I want the lowest 20% variances, this will give me the index of the variance, which value is at 20% of all variances)
    new_threshold=variance_list[new_threshold_idx] # this will give me the new threshold.
    variance_object.threshold=new_threshold
    
    return variance_object

def compare_values_in_dict(target_corr_dict,highest_corr_pair):
    """
    _________________________________________________________________________
    Returns the lower correlated to the target feature. To be used as part of   
        the greedy elimination of correlated pairs.
    _________________________________________________________________________
    Args:
        target_corr_dict(dict): dictionary containing the correlation coef
            between the target and the feature.
        highest_corr_pair (str): Indices of two features separated by "."
    
    Returns:
        the lower correlated feature to the target. If they are equal, returns
            the second one.

    """
    highest_corr_pair=str.split(highest_corr_pair,'.')
    target0=target_corr_dict[highest_corr_pair[0]]
    target1=target_corr_dict[highest_corr_pair[1]]
    
    if target0==target1:
        return highest_corr_pair[1] #to remove
    elif target0<target1:
        return highest_corr_pair[0]
    else:
        return highest_corr_pair[1]

def remove_correlated_features(X, y, combination_index, thresh=0.8, met='elimination'):
    """
    _________________________________________________________________________
    Remove correlated features using greedy elimination vs. greedy modularity 
        maximization  approaches.
    _________________________________________________________________________
    Args:
        X(np.array): 2D matrix of features
        y(np.array): target 1D matrix
        combination_index: 1D matrix of all features indices.
        thresh (float): the threshold for correlation. Default at 0.8
        met (str): To use greedy 'elimination' (Default) or greedy modularity
            'maximization'. 
                - 'elimination': will reiterively pick the highest correlated  
                feature pair and retain the feature that has the higher 
                correlation to the target. This process is repeated until there 
                is no other correlated pairs.
                - 'maximization': will find optimal communities in graph of 
                correlated features using Clauset-Newman-Moore greedy 
                modularity maximization. The node with the highest weighted 
                degree (weighted sum of all the connecting edges) in each 
                community will be used as the representative for that community.
                See more help(networkx.algorithms.community.
                greedy_modularity_communities)
    
    Returns:
        combination_index (np.array): set of new feature 
            indices (in the same order as the original set)
    """
    correlated_matrix=data_preprocessing.lower_triangle(abs(np.corrcoef(X,rowvar=False))) #extract the lower triangle of the absolute correlation matrix. this will have a shape (n_features^2)/2
    
    correlated_to_target=np.asarray([data_preprocessing.lower_triangle(abs(np.corrcoef(X[:,i],y,rowvar=False)))[0] for i in range(X.shape[1])]) #this will calculates the absolute correlation between each feature with the target. this will have shape (n_features,)
    
    combination_index_in_string=[str(i) for i in combination_index[::-1]]#the same combination_index in string format. Running with itertools.combinations will improve the timing.
    
    correlation_matrix_idx=[s1+"."+s2 for s1,s2 in combinations(combination_index_in_string,2)]#this create a list of correlated pairs index
    correlation_matrix_idx=np.asarray(correlation_matrix_idx)[::-1]# this will put them in the same order as the calculated correlated pairs. This will produce the lower triangle.

    correlated_pairs_idx=np.where(correlated_matrix>thresh)[0]#which pair is highly correlated. threshold of the corr coef is set at 0.8 Default
    
    temp_dict=dict(zip(correlation_matrix_idx[correlated_pairs_idx],correlated_matrix[correlated_pairs_idx])) #this will create a dictionary, where the keys are the correlated pairs, and the values are the highest correlated coefficient.
     
    temp_target_dict=dict(zip(map(str,combination_index),(correlated_to_target))) # this will create a dictionary, where the keys are the name of the combination feature, and the values is the corr coef to the target.
    
    if (met=='elimination'): #if greedy elimination is selected.
        
        features_to_be_removed=[] # this is the list of features that do not survive the greedy elimination, i.e. they have high corr coef to other features, but not high enough to the target.
        
        while len(temp_dict)!=0: # this will remove features until no high correlated pair is left.
            
            highest_correlation_pairs=max(temp_dict.items(),key=operator.itemgetter(1))[0]# highest correlation pairs from the correlation pair dictionary
            feature_to_remove=compare_values_in_dict(temp_target_dict,highest_correlation_pairs)
            new_temp_dict=dict(temp_dict) #create a new copy of the correlation pair dictionary that will update the temp_dict after each run
            
            for keys,values in temp_dict.items():
                temp_keys=str.split(keys,'.')
                if feature_to_remove in temp_keys:
                    features_to_be_removed.append(feature_to_remove)
                    del new_temp_dict[keys]
            temp_dict=new_temp_dict
        
        features_to_be_removed=np.asarray(list(set(features_to_be_removed))).astype('int64')
        
        combination_index=np.asarray([i for i in combination_index if i not in features_to_be_removed])
        
        return combination_index
    
    else: # using modularity maximization
        
        Graph=nx.MultiGraph()#this is an undirected graph
        source_list=[]
        target_list=[]
        edge_list=[str.split(i,'.') for i in temp_dict.keys()]#this is the list of edges
        for edge in edge_list:
            source_list.append(edge[0])
            target_list.append(edge[1])
        edges_df=pd.DataFrame(
            {
                "source":source_list,
                "target":target_list,
                "weight":[i for i in temp_dict.values()]
            }
        )
        Graph=nx.from_pandas_edgelist(edges_df,edge_attr=True)#create a graph
        
        s_graph=[Graph.subgraph(n) for n in greedy_modularity_communities(Graph)] # create number of subgraphs of connected components. Some of the graphs will have more than 100 features, and some will be only 2.
        
        to_retain_nodes=map(lambda x: max(dict(x.degree(weight='weight')).items(),key=operator.itemgetter(1))[0],s_graph)# this returns the nodes with highest weighted degree.

        Graph.remove_nodes_from(to_retain_nodes)
        features_to_be_removed=list(Graph.nodes)# the rest is to be removed.

        combination_index=np.asarray([i for i in combination_index if str(i) not in features_to_be_removed])#remove all the features from the combination index.
        
        return combination_index

class Low_Variance_Remover(BaseEstimator,SelectorMixin):
    """
    This will handle the removal of low variances
    """
    def __init__(self,variance_percent=0):
        """
        Initialize the object:
        variance_percent (float): the percentage of the variance to be 
                removed.
        """
        self.variance_percent=variance_percent
    
    def fit(self,X,y=None):
        
        """
        ______________________________
        Fitting the the transformer.
        ______________________________
        Args:
            X: 2D dataset of features.
            y: 1D vector of the target.
        
        Return
            self.variance_transformer: VarianceThreshold() with new threshold
            self.corr_idx
        """
        self.variance_transformer=lowest_percent_variance(percent=self.variance_percent,variance_object=VarianceThreshold().fit(X))
        
        self.variance_idx=np.where(self.variance_transformer.get_support())[0]#get variance transformed feature idx.
        return self
    
    def transform(self,X,y=None):
        """
        ______________________________
        Transforming the dataset.
        ______________________________
        Args:
            X: 2D dataset of features.
            y(optional): 1D vector of the target.
        
        Return
            new_X: transformed X
        """
        new_X=X[:,self.variance_idx]
        return new_X

    def _get_support_mask(self):
        return self.variance_transformer.get_support()
        

class High_Corr_Remover(BaseEstimator,TransformerMixin):
    
    '''
    This will handle all the feature reduction transformation of the dataset.
    '''
    def __init__(self,thresh=0.8,met='elimination'):
        
        """
        Initialize the object
        Args:
            thresh (float): the correlation threshold to be removed.
            met (str): the method of removing correlated features. Greedy 
                'elimination' (Default) or greedy modularity 'maximization'.
                
        Attributes:
            self.thresh
            self.met
        """
        
        self.thresh=thresh
        self.met=met
    
    def fit (self,X,y):
        """
        ______________________________
        Fitting the the transformer.
        ______________________________
        Args:
            X: 2D dataset of features.
            y: 1D vector of the target.
        
        Return
            self.corr_idx
        """
                 
        self.corr_idx=remove_correlated_features(X=X,y=y,combination_index=np.arange(X.shape[1]),thresh=self.thresh,met=self.met)
        
        return self
    
            
    def transform(self,X,y=None):
        """
        ______________________________
        Transforming the dataset.
        ______________________________
        Args:
            X: 2D dataset of features.
            y(optional): 1D vector of the target.
        
        Return
            new_X: transformed X
        """
        
        new_X=X[:,self.corr_idx]
        return new_X
