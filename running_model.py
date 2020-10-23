# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression,LogisticRegression
# from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,cross_val_score,GridSearchCV,cross_validate, cross_val_predict,StratifiedKFold
# from sklearn.metrics import accuracy_score, r2_score, mean_squared_error,make_scorer,confusion_matrix,multilabel_confusion_matrix

#Scikit-lib
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold,SelectFromModel


import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


import preprocessing
import pandas as pd
import numpy as np
import inspect
import operator
from itertools import chain


def lowest_percent_variance(percent,variance_object):
    """
    percent in the form of decimal points
    variance_object=VarianceThreshold().fit()
    if percent =0, then the threshold=0.0
    """
    variance_list=np.unique(variance_object.variances_) #this will give a sorted list of all feature variances
    new_threshold_idx=int(np.ceil(len(variance_list)*percent)) # this will give me the index of the new threshold (so if there are 100 unique variances and I want the lowest 20% variances, this will give me the index of the variance, which value is at 20% of all variances)
    new_threshold=variance_list[new_threshold_idx] # this will give me the new threshold.
    variance_object.threshold=new_threshold
    return variance_object

def compare_values_in_dict(target_corr_dict,highest_corr_pair):
    """
    Given a dictionary of feature and target and a list of the highest correlated pair, return the lower correlated key to be removed
    -----to be used as part of the greedy corr elimination-----
    Input: target_corr_dict= dictionary containing the correlation coef between the target and the feature.
    highest_corr_pair= the highest correlated pair
    Output:
    the lower correlated feature to the target
    """
    highest_corr_pair=str.split(highest_corr_pair,'.')
    target0=target_corr_dict[int(highest_corr_pair[0])]
    target1=target_corr_dict[int(highest_corr_pair[1])]
    
    if target0==target1:
        return highest_corr_pair[1] #to remove
    elif target0<target1:
        return highest_corr_pair[0]
    else:
        return highest_corr_pair[1]

def remove_correlated_features(X, y, combination_index, thresh=0.8, greedy=True):
    """
    Remove correlated features using greedy vs. PCA approaches
    Input
    X: 2D matrix of features
    y: target 1D matrix
    If greedy (Default) elimination: will reiterively pick the highest correlated feature pair and then retain the feature that has higher correlation to the target, and will remove the other one. This process is repeated until there is no correlated pairs.
    
    If PCA: Instead of eliminating recursively the correlated pairs, I look at the connected components in the correlated matrix. For each subgraph (connected network of correlated features) I perform greedy modularity maximization. Essentially I group the components in the subgraph into smaller communities, on which I perform PCA.
    """
    correlated_matrix=preprocessing.lower_triangle(abs(np.corrcoef(X,rowvar=False)),side_of_the_square=X.shape[1]) #extract the lower triangle of the absolute correlation matrix. this will have a shape (n_features^2)/2
    
    correlated_to_target=np.asarray([preprocessing.lower_triangle(abs(np.corrcoef(X[:,i],y,rowvar=False)),2)[0] for i in range(X.shape[1])]) #this will calculates the absolute correlation between each feature with the target. this will have shape (n_features,)
    
    correlation_matrix_idx=[]
    for i in combination_index:
        indices=[str(i)+'.'+str(n) for n in combination_index]#so 2.0 would mean the correlation between SFGdor.L_PreCG.R and PreCG.R_PreCG.L
        correlation_matrix_idx.append(indices)
        
    correlation_matrix_idx=np.vstack(correlation_matrix_idx)
    correlation_matrix_idx=preprocessing.lower_triangle(correlation_matrix_idx,side_of_the_square=correlation_matrix_idx.shape[0]) # this will let me know the name of the correlation pairs. The indices will have the same order of the original indices.

    correlated_pairs_idx=np.where(correlated_matrix>thresh)[0]#which pair is highly correlated. threshold of the corr coef is set at 0.8 Default
    
    
    temp_dict=dict(zip(correlation_matrix_idx[correlated_pairs_idx],correlated_matrix[correlated_pairs_idx])) #this will create a dictionary, where the keys are the correlated pairs, and the values are the highest correlated coefficient.
     
    temp_target_dict=dict(zip(combination_index,(correlated_to_target))) # this will create a dictionary, where the keys are the name of the combination feature, and the values is the corr coef to the target.
    
    if greedy: #if greedy elimination is selected.
        
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
    
    else: #using PCA
        
        Graph=nx.MultiGraph()#this is undirected graph
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
            }
        )
        Graph=nx.from_pandas_edgelist(edges_df)#create a graph
                
                
        s_graph=[Graph.subgraph(n) for n in nx.connected_components(Graph)] # create number of subgraphs of connected components. Some of the graphs will have more than 100 features, and some will be only 2.


        PCA_list=[]
        #create a list of features combinations, that will be passed to PCA.
        for small_graph in s_graph:
            #pass in greedy_modularity_communities:
            com=list(greedy_modularity_communities(small_graph))
            com=[sorted(c) for c in com]
            PCA_list.append(com)

        PCA_list=list(chain.from_iterable(PCA_list)) # to remove one double square bracket

        features_to_be_removed=list(chain.from_iterable(PCA_list))

        print('len of combination_index is:%d'%len(combination_index))

        combination_index=np.asarray([i for i in combination_index if str(i) not in features_to_be_removed])#remove all the features from the combination index.

        PCA_list=[sorted([int(i) for i in list(n)]) for n in PCA_list]# makes sure that the features passed into PCA are sorted. So that later on the feature names will be the same.

        #return combination_index, s_graph
        print('len of combination_index is:%d'%len(combination_index))
        
        return combination_index,PCA_list

class FeatureReduction(BaseEstimator,TransformerMixin):
    
    """
    This will handle all the transformation of the data
    Parameters:
    variance_percent= denotes the percentage of features removed by the VarianceThreshold. 0.20 (Default) denotes to remove the lowest 20% of variances. See running_model.lowest_percent_variance for more information
    variant_transformer=VarianceThreshold Object.
    
    
    """    
    def __init__(self,variance_percent=.2,variance_transformer=VarianceThreshold(),thresh=0.8,greedy=True):
        
        args, _, _, values=inspect.getargvalues(inspect.currentframe())
        values.pop('self')
        
        for arg,val in values.items():
            setattr(self, arg,val)
    
    def fit (self,X,y):
        
        self.variance_transformer=lowest_percent_variance(percent=self.variance_percent,variance_object=self.variance_transformer.fit(X))
        
        variance_idx=np.where(self.variance_transformer.get_support())[0]#get variance transformed feature idx.
        
        if self.greedy:
            
            self.corr_idx=remove_correlated_features(X=X[:,variance_idx],y=y,combination_index=variance_idx,thresh=self.thresh,greedy=self.greedy)
            
        else:
            
            self.corr_idx,self.PCA_list=remove_correlated_features(X=X[:,variance_idx],y=y,combination_index=variance_idx,thresh=self.thresh,greedy=self.greedy)
        
            self.pca_dict=dict()
            self.pca_dict_name=dict()
            
            for i in range(len(self.PCA_list)):
                self.pca_dict[i]=PCA(n_components=0.8,random_state=42).fit(X[:,self.PCA_list[i]])
                self.pca_dict_name[i]=['_'.join(['PCA','.'.join([str(f) for f in self.PCA_list[i]]),str(n)]) for n in range(len(self.pca_dict[i].explained_variance_ratio_))]
                
        return self
    
            
    def transform(self,X,y=None):
        if self.greedy:
            new_X=X[:,self.corr_idx]
            return new_X
        
        else:
            
            new_X=X[:,self.corr_idx]
            print('shape of new_X is:', new_X.shape)
            for i in range(len(self.pca_dict)):
                new_X=np.append(new_X,self.pca_dict[i].transform(X[:,self.PCA_list[i]]),axis=1)    
            print('shape of new_X is:', new_X.shape)
            return new_X



# class scikit_model:
#     """
#     docstring
#     """
#     def __init__(self,model,X,y,hyperparameters,filepath=None,model_name=None,step=0.001,grid_search=True):
#         """
#         """
#         self.model=model
        
#     def Select_From_Model(self):
#         """
#         docstring
#         """
#         inner_cv=StratifiedKFold(n_splits=4,random_state=42)
#         outer_cv=StratifiedKFold(n_splits=5,random_state=42)
        
#         fold_number=0
        
#         for train_index,test_index in outer_cv.split(self.X,self.y):
            
#             fold_number+=1
#             #divide training sets into outer folds
            
#             X_train=self.X[train_index,:]
#             y_train=self.y[train_index]
#             X_test=self.X[test_index,:]
#             y_test=self.y[test_index]

            
            
        
# def Select_From_Model(model,hyperparameter,filepath=None,model_name=None,step=0.001,grid_search=True,rfe_cv=False,Features_corrected=Features_corrected,
#            data_Status=data_Status):

#     test_sensitivity=[]
#     test_specificity=[]
#     test_accuracy=[]
#     cross_validate_sensitivity=[]
#     cross_validate_specificity=[]
#     cross_validate_accuracy=[]
#     n_features=[]
#     inner_cv=StratifiedKFold(n_splits=4,random_state=42)
#     outer_cv=StratifiedKFold(n_splits=5,random_state=42)

#     fold_number=0

#     for train_index,test_index in outer_cv.split(Features_corrected,data_Status):

#         fold_number+=1
#         #divide training sets into outer folds
#         X_train=Features_corrected[train_index,:]
#         y_train=data_Status[train_index]
#         X_test=Features_corrected[test_index,:]
#         y_test=data_Status[test_index]

#         #Standardscale the dataset based on the X_train set and correct for imbalance in training dataset
#         scaler=StandardScaler()
#         scaler.fit(X_train)
#         X_train_scaled=scaler.transform(X_train)
#         X_test_scaled=scaler.transform(X_test)

#         sfm=SelectFromModel(model)
#         sfm.fit(X_train_scaled,y_train)    

#         if grid_search==True:
#             clf = GridSearchCV(estimator=model, param_grid=hyperparameter, cv=inner_cv,
#                                    iid=False)
#             clf.fit(sfm.transform(X_train_scaled),y_train)
#             tuned_model=clf.best_estimator_.fit(sfm.transform(X_train_scaled),y_train)
#         else:
#             tuned_model=model.fit(sfm.transform(X_train_scaled),y_train)
        
#         if rfe_cv==True:
#             rfe = RFECV(estimator=tuned_model, step=step, cv=inner_cv,verbose=0)
#             rfe.fit(sfm.transform(X_train_scaled),y_train)
            
# #             pickle_dump(rfe,filepath=filepath,model_name=model_name,
# #                        fold_number=fold_number)
            
#             tuned_model=rfe
        
#         scores=cross_validate(tuned_model,sfm.transform(X_train_scaled),y_train,scoring=scoring,cv=inner_cv)

#         cross_validate_sensitivity.append(scores['test_sensitivity'].mean())
#         cross_validate_specificity.append(scores['test_specificity'].mean())  
#         cross_validate_accuracy.append(scores['test_accuracy'].mean())


#         y_pred=tuned_model.predict(sfm.transform(X_test_scaled))
#     #    my test scores append them based on folds.
#         test_sensitivity.append(sensitivity(y_pred=y_pred,y_true=y_test))
#         test_specificity.append(specificity(y_pred=y_pred,y_true=y_test))
#         test_accuracy.append(accuracy_score(y_pred=y_pred,y_true=y_test))
#     print(X_train_scaled.shape)
#     output=dict()
#     output['test_accuracy']=test_accuracy
#     output['test_sensitivity']=test_sensitivity
#     output['test_specificity']=test_specificity
#     output['cross_validate_sensitivity']=cross_validate_sensitivity
#     output['cross_validate_specificity']=cross_validate_specificity
#     output['cross_validate_accuracy']=cross_validate_accuracy
#     return output