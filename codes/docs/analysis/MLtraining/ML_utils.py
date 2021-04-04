"""ML_utils.py
This custom file contains functions to run scikit-learn preprocessing pipelines
    and training ML models.
"""
#Scikit-lib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold,SelectorMixin
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Network visualisation and algorithm
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

#Python essentials

import operator
from itertools import combinations
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
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

# def compare_values_in_dict(target_corr_dict,highest_corr_pair):
#     """
#     _________________________________________________________________________
#     Returns the lower correlated to the target feature. To be used as part of   
#         the greedy elimination of correlated pairs.
#     _________________________________________________________________________
#     Args:
#         target_corr_dict(dict): dictionary containing the correlation coef
#             between the target and the feature.
#         highest_corr_pair (str): Indices of two features separated by "."
    
#     Returns:
#         the lower correlated feature to the target. If they are equal, returns
#             the second one.

#     """
#     highest_corr_pair=str.split(highest_corr_pair,'.')
#     target0=target_corr_dict[highest_corr_pair[0]]
#     target1=target_corr_dict[highest_corr_pair[1]]
    
#     if target0==target1:
#         return highest_corr_pair[1] #to remove
#     elif target0<target1:
#         return highest_corr_pair[0]
#     else:
#         return highest_corr_pair[1]

# def remove_correlated_features(X, y, combination_index, thresh=0.8, met='elimination'):
#     """
#     _________________________________________________________________________
#     Remove correlated features using greedy elimination vs. greedy modularity 
#         maximization  approaches.
#     _________________________________________________________________________
#     Args:
#         X(np.array): 2D matrix of features
#         y(np.array): target 1D matrix
#         combination_index: 1D matrix of all features indices.
#         thresh (float): the threshold for correlation. Default at 0.8
#         met (str): To use greedy 'elimination' (Default) or greedy modularity
#             'maximization'. 
#                 - 'elimination': will reiterively pick the highest correlated  
#                 feature pair and retain the feature that has the higher 
#                 correlation to the target. This process is repeated until there 
#                 is no other correlated pairs.
#                 - 'maximization': will find optimal communities in graph of 
#                 correlated features using Clauset-Newman-Moore greedy 
#                 modularity maximization. The node with the highest weighted 
#                 degree (weighted sum of all the connecting edges) in each 
#                 community will be used as the representative for that community.
#                 See more help(networkx.algorithms.community.
#                 greedy_modularity_communities)
    
#     Returns:
#         combination_index (np.array): set of new feature 
#             indices (in the same order as the original set)
#     """
#     correlated_matrix=data_preprocessing.lower_triangle(abs(np.corrcoef(X,rowvar=False))) #extract the lower triangle of the absolute correlation matrix. this will have a shape (n_features^2)/2
    
#     correlated_to_target=np.asarray([data_preprocessing.lower_triangle(abs(np.corrcoef(X[:,i],y,rowvar=False)))[0] for i in range(X.shape[1])]) #this will calculates the absolute correlation between each feature with the target. this will have shape (n_features,)
    
#     combination_index_in_string=[str(i) for i in combination_index[::-1]]#the same combination_index in string format. Running with itertools.combinations will improve the timing.
    
#     correlation_matrix_idx=[s1+"."+s2 for s1,s2 in combinations(combination_index_in_string,2)]#this create a list of correlated pairs index
#     correlation_matrix_idx=np.asarray(correlation_matrix_idx)[::-1]# this will put them in the same order as the calculated correlated pairs. This will produce the lower triangle.

#     correlated_pairs_idx=np.where(correlated_matrix>thresh)[0]#which pair is highly correlated. threshold of the corr coef is set at 0.8 Default
    
#     temp_dict=dict(zip(correlation_matrix_idx[correlated_pairs_idx],correlated_matrix[correlated_pairs_idx])) #this will create a dictionary, where the keys are the correlated pairs, and the values are the highest correlated coefficient.
     
#     temp_target_dict=dict(zip(map(str,combination_index),(correlated_to_target))) # this will create a dictionary, where the keys are the name of the combination feature, and the values is the corr coef to the target.
    
#     if (met=='elimination'): #if greedy elimination is selected.
        
#         features_to_be_removed=[] # this is the list of features that do not survive the greedy elimination, i.e. they have high corr coef to other features, but not high enough to the target.
        
#         while len(temp_dict)!=0: # this will remove features until no high correlated pair is left.
            
#             highest_correlation_pairs=max(temp_dict.items(),key=operator.itemgetter(1))[0]# highest correlation pairs from the correlation pair dictionary
#             feature_to_remove=compare_values_in_dict(temp_target_dict,highest_correlation_pairs)
#             new_temp_dict=dict(temp_dict) #create a new copy of the correlation pair dictionary that will update the temp_dict after each run
            
#             for keys,values in temp_dict.items():
#                 temp_keys=str.split(keys,'.')
#                 if feature_to_remove in temp_keys:
#                     features_to_be_removed.append(feature_to_remove)
#                     del new_temp_dict[keys]
#             temp_dict=new_temp_dict
        
#         features_to_be_removed=np.asarray(list(set(features_to_be_removed))).astype('int64')
        
#         combination_index=np.asarray([i for i in combination_index if i not in features_to_be_removed])
        
#         return combination_index
    
#     else: # using modularity maximization
        
#         Graph=nx.MultiGraph()#this is an undirected graph
#         source_list=[]
#         target_list=[]
#         edge_list=[str.split(i,'.') for i in temp_dict.keys()]#this is the list of edges
#         for edge in edge_list:
#             source_list.append(edge[0])
#             target_list.append(edge[1])
#         edges_df=pd.DataFrame(
#             {
#                 "source":source_list,
#                 "target":target_list,
#                 "weight":[i for i in temp_dict.values()]
#             }
#         )
#         Graph=nx.from_pandas_edgelist(edges_df,edge_attr=True)#create a graph
        
#         s_graph=[Graph.subgraph(n) for n in greedy_modularity_communities(Graph)] # create number of subgraphs of connected components. Some of the graphs will have more than 100 features, and some will be only 2.
        
#         to_retain_nodes=map(lambda x: max(dict(x.degree(weight='weight')).items(),key=operator.itemgetter(1))[0],s_graph)# this returns the nodes with highest weighted degree.

#         Graph.remove_nodes_from(to_retain_nodes)
#         features_to_be_removed=list(Graph.nodes)# the rest is to be removed.

#         combination_index=np.asarray([i for i in combination_index if str(i) not in features_to_be_removed])#remove all the features from the combination index.
        
#         return combination_index

def remove_correlated_features(X,target,threshold=0.8):
    """
    Remove correlated features.
    Args:
        X np.array: the feature matrix.
        target np.array: 1D shape the target that the features are selected 
        upon.
        threshold (0-1): correlation coefficient threshold the correlated pairs 
        are selected.
    
    Returns:
        dict_of_features (defaultdict(list)): dictionary where the keys are the
        index of the features, and the values, are the index of features that 
        the key feature represent.
    
    Example:
        > remove_correlated_features(X,target,threshold=0.8)
        >> {[0]: [0],[1]: [1,2,3,4],...}
        here the feature 1 can represent the features 2,3 and 4.
    
    """
    X=X.copy() # avoid changes to the original array
    dict_of_features=defaultdict(list)
    for i in range(X.shape[1]):
        dict_of_features[i].append(i)
    
    while True:
        corr_matrix_between_features=np.corrcoef(X,rowvar=False) # calculate the correlation matrix between features
        corr_matrix_between_features[np.isnan(corr_matrix_between_features)]=0 # set any nan number to 0.
        np.fill_diagonal(corr_matrix_between_features,0) # fill the diagonal of 1s to 0s.
        maximum_correlation=np.max(np.abs(corr_matrix_between_features))
        if maximum_correlation < threshold:
            break
        # for some reason, when the nrow (X) is < ncol(X), then the resulting correlation matrix is symmetrical, but not when nrow(X) is less than ncol(X), the difference in the non-symmetrical is very small, almost non-negligible. However, when you run np.where, this will result in differing output.
        try:
            feature_0,feature_1=np.where(np.abs(corr_matrix_between_features)==maximum_correlation)[0] # find where the highest correlation pair is
        except ValueError: # when the unpacking is unsuccessful, feature is an array int
            feature_0,feature_1=np.where(np.abs(corr_matrix_between_features)==maximum_correlation)
            feature_0=feature_0[0] #get it into an interger format to be consistent
            feature_1=feature_1[0]
        feature_0_corr_coef=pearsonr(X[:,feature_0],target)[0]
        feature_1_corr_coef=pearsonr(X[:,feature_1],target)[0]
        if feature_0_corr_coef>feature_1_corr_coef: # check which one has higher correlation to the target
            dict_of_features[feature_0].extend(dict_of_features[feature_1]) # represent the lower correlated feature with the higher one.
            X[:,feature_1]=np.nan # set the values in the eliminated feature as nan. so that next time, we don't have to recompute the correlation.
            del dict_of_features[feature_1] # delete that key
        else:
            dict_of_features[feature_1].extend(dict_of_features[feature_0])
            X[:,feature_0]=np.nan
            del dict_of_features[feature_0]

    return dict_of_features

def merge_features_values(X,dict_of_features,average=False):
    """
    Merge the correlated features, either by elimination or averaging
    Arg:
        X: np.array matrix of features
        dict_of_features (defaultdict(list)): dictionary where the keys are the
        index of the features, and the values, are the index of features that 
        the key feature represent.
        average (bool): if False(default) then the features are not merged, but 
        only the most correlated feature is retained.
    
    Output:
        X: np.array matrix of features of the same shape as the original. But 
        the removed feature columns are transformed to np.nan.
    
    Example:
        The outputed X array can be plugged to pandas Dataframe where the 
        columns have been named. The Dataframe can dropna(axis=1) to remove the 
        correlated features.
        
    """
    X=X.copy()
    if average:
        for key,values in dict_of_features.items():
            if len(values)>1:
                X[:,key]=np.mean(X[:,values],axis=1)

    for col in range(X.shape[1]):
        if not dict_of_features[col]:#if this key is empty
            X[:,col]=np.nan
    return X  

def merge_features_names(dict_of_features,feature_names):
    """
    Used in conjunction with the merge the features to get the name of the 
    representative feature name and its constituents.
    Args:
        dict_of_features: dictionary where the keys are the
        index of the features, and the values, are the index of features that 
        the key feature represent.
        feature_names: pandas.Dataframe.columns or a list of names.
    
    Output:
        new_dict_of_features (defaultdict(list)): same as dict_of_features, but
        the indices are transformed to string names.
    
    Example:
        defaultdict(list,
            {'SFGdor.R_PreCG.L': ['SMA.R_PreCG.L'],
             'ORBsup.L_SFGdor.R': ['SFGmed.R_ORBsup.L'],
             'ORBsup.R_PreCG.R': ['ORBmid.R_PreCG.R'],...}
    """
    new_dict_of_names=defaultdict(list)
    for key,values in dict_of_features.items():
        if len(values)>1:
            new_dict_of_names[feature_names[key]]=list(feature_names[values[1:]])
    return new_dict_of_names


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

class Retain_non_zero_features(BaseEstimator,TransformerMixin):
    """
    Removes features that register zero values across multiple subjects.
    Functions:
        __init__(self,perc_threshold): set the percentage of zero allowed.
        fit(self,X,y=None): calculate the percentage of zeros values across 
            each features
        transform(self,X,y=None): returns the transformed feature matrix.
        get_column_names(self,feature_names): get the name of the retained 
            features.
    Attributes:
        @self.perc_threshold= the percentage of zero allowed (0-1)
        @self.zero_perc= calculated percentages of zero across features columns
        @self.reduced_feature_names= array of the retained feature names.
    """
    def __init__(self,perc_threshold=0):
        self.perc_threshold=perc_threshold
    
    def fit(self,X,y=None):
        zero_perc_calc=lambda feature: np.sum(feature==0)/len(feature)
        self.zero_perc=np.apply_along_axis(zero_perc_calc,0,X) #perform this func across the column axis.
        
        return self
    
    def transform(self,X,y=None):
        new_X=X.copy()
        new_X=new_X[:,~(self.zero_perc>self.perc_threshold)]
        return new_X
    
    def get_column_names(self,feature_names):
        self.reduced_feature_names=feature_names[~(self.zero_perc>self.perc_threshold)]
        
        return self
        
        

class High_Corr_Remover(BaseEstimator,TransformerMixin):
    
    '''
    This will handle all the feature reduction transformation of the dataset.
    Functions:
        __init__(self,target,thresh,average)
        fit(self,X,y=None): fit the 2D feature matrix to the transformer.
        > Returns a feature_dict, which contains the original features' index 
            as a key, and values as the list of indices that the key represent.
        transform(self,X,y=None)
        > Returns the reduced new_X matrix.
    Attributes:
        @self.features_dict=dictionary, where keys are indices of all features, 
        and the values are the indices of all features that the key represents.
        @self.features_dict_strings= reduced dictionary of the features_dict, 
        where only the keys, with the values length more than 1 is included. 
        The keys and values are mapped to strings.
    
    '''
    def __init__(self,target,thresh=0.8,average=False):
        """
        Initialise the model with the target matrix
        Arg:
            target (1D np.array): the target to which the features' correlation 
            matrix is calculated. (E.g. the PRS score)
            thresh (0-1): the correlation coefficient threshold.
            average (bool): whether to merge by averaging.

        """
        self.target=target
        self.thresh=thresh
        self.average=average

    
    def fit (self,X,y=None):
        """
        ______________________________
        Fitting the the transformer.
        ______________________________
        Args:
            X: 2D dataset of features.
            y: 1D vector of the target.
        
        Return
            self.features_dict (defaultdict(list)): {[0]:[0],[1]:[1,2,3]...}
        """
                 
        self.features_dict=remove_correlated_features(X,self.target,self.thresh)

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
        
        new_X=merge_features_values(X,self.features_dict,self.average)
        
        new_X=new_X[:,~np.isnan(new_X).any(axis=0)] #remove the nan columns
        return new_X
    
    def get_column_names(self,feature_names):
        """
        feature_names: the list of features as strings. (e.g. pd.Dataframe.columns)
        Returns:
            feature_dict_strings (defaultdict): the key are the features, that 
            represent other features. This dictionary does not include the 
            names of features, that do not represent other features.
        """
        
        self.feature_dict_strings=merge_features_names(self.features_dict,feature_names)
        return self

    
def print_scores(y_pred,y_true):
    return {'r2': r2_score(y_true,y_pred),
    'MAE': mean_absolute_error(y_true,y_pred),
    'MSE': mean_squared_error(y_true,y_pred),
    'Correlation':pearsonr(y_true,y_pred)[0],
    'p_value':pearsonr(y_true,y_pred)[1]}

def splitting_dataset_into_k_fold(X,y,k=5):
    """
    For nested_CV, divide the dataset into inner and outer folds
    Args:
        X= dataset
        y= label
        k= folds number
    return
        generator of format
            (X_train,y_train,X_test, y_test)
    """
    outer_cv=KFold(n_splits=k)
    for trainval_index,test_index in outer_cv.split(X,y):
        X_trainval=X[trainval_index,:]
        y_trainval=y[trainval_index]
        X_test=X[test_index,:]
        y_test=y[test_index]
        yield (X_trainval,y_trainval,X_test,y_test)
        
def get_model_bias(y_true, y_pred):
    """
    Correct for bias in the regression model as shown in Yassine et al., 2020, 
    by fitting the predicted values to the true values in another linear model.
    The updated unbiased predicted values equals to the original predicted values * coeff + intercept
    Args:
        y_true
        y_pred
    Returns:
        model Coeff
        Intercept
    """
    model_error=y_true - y_pred
    lin_reg=LinearRegression()
    lin_reg.fit(y_true.reshape(-1,1),model_error)
    coeff=lin_reg.coef_
    intercept=lin_reg.intercept_
    return coeff,intercept