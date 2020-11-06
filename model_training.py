"""
Executing the finetuning and model selection. This is used for regression 
    model, but can be modified to fit other problems.
"""
import warnings
warnings.simplefilter(action='ignore', category=Warning)


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_regression, RFECV, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.inspection import permutation_importance

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import joblib #to save models

import inspect
import running_model

import numpy as np
import sys
    
import pickle
import operator


def save_a_model(model, model_name, split_no, filepath):
    """
    ___________________________
    Save the model externally
    ___________________________
    Args:
        model (scikit-model object)
        model_name(str): name of the model
        split_no(int): split number
        filepath(str): filepath
    
    Return:
        saves the model externally
    """
    filepath = filepath+'split_'+str(split_no)+model_name+'.pkl'
    return joblib.dump(model, filepath)

def save_a_npy(npy_array, npy_name, split_no, filepath):
    """
    ______________________________________________________
    Save the numpy array in binary format externally
    ______________________________________________________
    Args:
        npy_array (numpy array)
        npy_name(str): name of the array
        split_no(int): split number
        filepath(str): filepath
    
    Return:
        saves the array in binary format externally
    """
    filepath = filepath+'split_'+str(split_no)+npy_name+'.npy'
    return np.save(filepath, npy_array)

def save_the_object(object, filepath):
    
    with open(filepath, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)

def load_the_object(filepath):
    with open(filepath, 'rb') as input:
        x = pickle.load(input)
    return x

def get_permutation_importances(model, X, y, scoring='r2'):
    
    """
    __________________________________________________________________________
    Get permutation importance for each feature. And then return the ones with 
        the importances above the mean.
    ___________________________________________________________________________
    First a baseline metric is caculated from the data X. Then the feature 
        columns are permuted and the metric is calculated again. The 
        permutation importance is difference between the baseline metrics and 
        metric permutating the feature columns
    
    Args:
        model (scikit model):
        X (np.array):
        y (np.array):
        scoring (str): default 'r2'
        
    Return:
        indices: the indices of the feature importances.
    """
    
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=10, random_state=42)
    
    mean_threshold = np.mean(result.importances_mean)

    indices = np.where(result.importances_mean >= mean_threshold)[0]

    return indices


def fine_tune_hyperparameters(param_dict, model, X, y, model_name, fine_tune='grid', cv=4, scoring='r2'):
    """
    ______________________________________________________
    Manual Fine tuning of hyperparameters
    ______________________________________________________
    Args:
        param_dict=dict({'lin_reg':None,
                         'lasso':{'alpha':[0]},
                         'ridge':{'alpha':np.linspace(200,500,10)},
                         'random_forest':{'n_estimators':[3,10,30],
                         'max_features':[2,4,6,8]},
                         'svm':{},
                         'knn':{}})

                         
        model (scikit_model): scikit model
        X (np.array): the dataset
        y (1D np.array): the label target
        model_name (str): name corresponding to the param_dict.keys()
        fine_tune (str): using either 'grid' or 'randomized'
        cv (KFold or int)= cross-validation partition.
        scoring (str): scoring metric used
        
    
    Return:
        
        search.best_estimator_: best estimator from the fine_tuning process
    """
    examined_param_grid={} #this will be the dictionary that contains the past examined parameter grids
    model_examined=0
    while True:
        if model_examined!=0:
            print('past parameters examined:\n')
            n=[print(i) for i in [(key,value['search_best_params'],value['best_score']) for key,value in examined_param_grid.items()]]
            
        print('make sure you write as if you are writing a command, i.e. with square bracket for list')
        try:
            x=[eval(input(i)) for i in param_dict[model_name].keys()]#this will take the input from the user. make sure to type [1,2,3,4] (i.e. with square brackets)
            
        except SyntaxError:
            print('try again')
            continue

        except AttributeError:
            print('There is no parameters to fine_tune, if this is not the case, please add parameters to the model_training.parameters_dict if that is not correct \n')
            model.fit(X,y)
            return model
            break

        param_grid=dict(zip(param_dict[model_name].keys(),x)) 
        model_examined+=1
        examined_param_grid[model_examined]={}#this will create a dynamic nested dicionary
        examined_param_grid[model_examined]['param_grid']=param_grid
        
        if fine_tune=='grid':
            search=GridSearchCV(model,param_grid=param_grid,cv=cv,iid=False,scoring=scoring)
            
        else:
            search=RandomizedSearchCV(model,param_distributions=param_grid,cv=cv,n_iter=100,scoring=scoring)
        
        search.fit(X,y)
        
        examined_param_grid[model_examined]['search_model_best_estimator']=search.best_estimator_# save into that dictionary the best_estimator, best_parameter and best scores.
        examined_param_grid[model_examined]['search_best_params']=search.best_params_
        examined_param_grid[model_examined]['best_score']=np.sqrt(-search.best_score_)

        cvres=search.cv_results_
        
        print('here is the fine tuning results \n')
        for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
            print(np.sqrt(-mean_score),params)
        
        loop_continue=input('Do you want to repeat? (yes/no): ')
        if loop_continue=='no':
            print('which model you want to choose:')# choose from the list the best model
            n=[print(i) for i in [(key,value['search_best_params'],value['best_score']) for key,value in examined_param_grid.items()]]
            the_model_i_want=eval(input('select a model number'))
            return examined_param_grid[the_model_i_want]['search_model_best_estimator']
            break

def performing_rfecv(model,X,y,step,combination_idx,split_no,filepath,cv,scoring='r2'):
    """
    ___________________________________________________
    Performing scikit recursive feature elimination CV.
    ___________________________________________________
    Args:
        model (scikit-model): scikit object            
        X (np.asarray): Training dataset
        y (np.asarray): label target
        combination_idx (np.array): feature indices of the X.
        split_no (int):
        filepath (str):
        cv (cross-validated)
        scoring (str)
        
    Returns:
        rfecv (rfecv scikit object)
        rfecv_estimator: fitted rfecv estimator
        combination_idx_after_rfecv
        scores_after_rfecv
        
    ============================================================================
    Notes:
    
    Here, we implement the use of Recursive Feature Elimination. It utilises the underlying coef_ or feature_importance attributes of the model. It ranks the features by their estimated coefficients, and removes the weakest ones once the specified number of feature is reached. Cross-validation is used to score different feature subsets and select the best scoring collection of features. 
    
    ============================================================================
    """
    
    print('I am beginning the Recursive Feature Elimination')
    
    rfecv=RFECV(estimator=model,step=step,scoring=scoring,cv=cv).fit(X,y)
    
    combination_idx_after_rfecv=rfecv.transform(combination_idx)
    
    save_a_npy(combination_idx_after_rfecv,npy_name='combination_idx_after_rfecv',split_no=split_no,filepath=filepath)
    #save the new combination indices after the final filter.
    
    save_a_model(rfecv,model_name='rfecv',split_no=split_no,filepath=filepath)
    #save the rfecv model
    
    print('my RFECV is done for fold %d'%split_no)
    rfecv_estimator=rfecv.estimator
    rfecv_estimator.fit(rfecv.transform(X),y)
    scores_after_rfecv=cross_val_score(rfecv_estimator,rfecv.transform(X),y,scoring=scoring,cv=cv) #get the estimated performance scores
    
    return rfecv, rfecv_estimator, combination_idx_after_rfecv,scores_after_rfecv


def performing_sfscv(model,X,y,step,combination_idx,split_no,filepath,cv,scoring='r2'):
    
    """
    ___________________________________________________
    Performing mlxtend Sequential Feature Selector CV.
    ___________________________________________________
    Args:
        model (scikit-model): scikit object            
        X (np.asarray): Training dataset
        y (np.asarray): label target
        combination_idx (np.array): feature indices of the X.
        split_no (int):
        filepath (str):
        cv (cross-validated)
        scoring (str)
        
    Returns:
        sfscv (sfscv scikit object)
        sfscv_estimator: fitted sfscv estimator
        combination_idx_after_sfscv
        scores_after_sfscv
    ============================================================================
    Notes:
    
    Here, we implement the use of Sequential Feature Selector. 
    In a greedy fashion, we remove one feature at a time (forward=False), and 
    choose the subset that yields the best model with the best 
    scoring (scoring='r2') metrics in all cross-validation 
    splits. We also allow for conditional inclusion (floating=True), if any of 
    the removed feature if included back, can improve the model.
    Instead of setting a priori number of feature, we choose the subset that yileds the best score (k_features='best')
    
    ============================================================================
    
    
    """
    
    print('I am beginning the Sequential Feature Selector \n')
    
    sfscv=SFS(model,k_features='best',forward=False,floating=True,verbose=1,scoring=scoring,cv=cv,n_jobs=-1)
    sfscv.fit(X,y)
    
    combination_idx_after_sfscv=sfscv.transform(combination_idx)
    
    save_a_npy(combination_idx_after_sfscv,npy_name='combination_idx_after_sfscv',split_no=split_no,filepath=filepath)
    #save the new combination indices after the final filter.
    
    save_a_model(sfscv,model_name='sfscv',split_no=split_no,filepath=filepath)
    #save the sfscv model
    
    print('my SFSCV is done for fold %d'%split_no)
    sfscv_estimator=sfscv.estimator
    sfscv_estimator.fit(sfscv.transform(X),y)
    scores_after_sfscv=cross_val_score(sfscv_estimator,sfscv.transform(X),y,scoring=scoring,cv=cv) #get the estimated performance scores
    
    return sfscv, sfscv_estimator, combination_idx_after_sfscv,scores_after_sfscv

def get_the_best_model(X_test,y_test,filepath,fold_number,*args):
    """
    ___________________________
    args: best_model_key,best_model,cv_score,combination_idx_array
    
    CHECK IF FEATURE PRUNING IMPROVED THE MODEL
    ___________________________
    """
    
    best_model_key,best_model,cv_score,combination_idx_array=max(args,key=operator.itemgetter(2))
    with open(filepath+'log.txt','a+') as file:
        file.write('use the combination after %s for split_no %d \n'%(best_model_key,fold_number))
    
    y_pred=best_model.predict(X_test[:,combination_idx_array.reshape(-1)])
    model_r2=r2_score(y_test,y_pred)
    
    return model_r2


class scikit_model:
    """
    Handles the fine-tuning and feature selection.
    """

    def __init__(self,model,X,y,fine_tune='grid',filepath='./',model_name=None,step=1,random_state=42):
        
        """
        ___________________________
        Initialize the object
        ___________________________
        Args:
            model (scikit object): the scikit model
            X (np.array): the dataset
            y (np. array): the label in 1D vector
            fine_tune (str): 'grid'= GridSearchCV, 'random'= RandomSearchCV
            filepath(str):' the file path
            model_name(str): the name of the model
            step (int or float): the name of feature removed after each step in 
                recursive feature elimination CV.
            random_state (int): random state for cross-validation 
                train_test_split.
        
        Attributes:
            self.model
            self.X
            self.y
            filepath
            self.fine_tune (str): grid vs. randomized
            self.model_name
        
        Functions:
            feature_selection_model(self,combination_idx=np.arange(4005))
        """
        
        args, _, _, values=inspect.getargvalues(inspect.currentframe())
        values.pop('self')
        
        for arg,val in values.items():
            setattr(self, arg,val)
        
        self.parameters_dict=dict({'lin_reg':None,
                         'lasso':{'alpha':[0]},
                         'ridge':{'alpha':np.linspace(200,500,10)},
                         'random_forest':{'n_estimators':[3,10,30],'max_depth':[0],'min_samples_split':[0],'min_samples_leaf':[0],'max_leaf_nodes':[0]},
                         'lin_svr':{'C':[0],'epsilon':[0]},
                         'knn':{'n_neighbors':[int(i) for i in np.linspace(1,20,20)],'weights':['uniform','distance'],'leaf_size':[0],'metric':['minkowski','euclidean','manhattan'],'p':[0]}})

    def feature_selection_model(self,combination_idx=np.arange(4005),do_feature_pruning='both'):
        """
        ___________________________
        Feature selecting the model
        ___________________________
        
        First, the data is divided into 5 folds. 1 folds is saved as test set,
            and the others are for training and cross-validation.
        
        Second, Standard_scaler is applied to training sets, and transform the 
            tests sets.
        
        Third, FeatureReduction (VarianceThreshold and CorrelationThreshold) is 
            applied. The results were then passed to SelectPercentile 
            (percentile set at 20%) to select for the highest correlated to the 
            target features.
        
        Fourth, GridSearch (if fine_tune='grid') or RandomSearch is applied to 
            find the best hyperparameters. The results were passed to a 
            feature permutation importance to select for the most important 
            features.
            
        Fifth, the model is then passed to the recursive feature elimination or 
            Sequential Feature Selector, to prune the rest of the features. See 
            mlxtend.feature_selection.SequentialFeatureSelector for more 
            information. (sklearn v. 0.24 will start implementing this function)
        
        Args:
            combination_idx= the features idx.
            do_feature_pruning (str): 'both' : do both Recursive Feature 
            elimination and SequentialFeatureSelector. Output CV score for both 
            and compare them with feature permutation importance step.
            'rfecv': Do only rfecv or 'sfs': do only sfs.
            'none' for neither.
        
        Returns:
            saves externally the combination_idx 
            self.cross_validated_scores_after_rfecv
            self.test_scores_across_all_split
        """
        inner_cv=KFold(n_splits=4,random_state=self.random_state)
        outer_cv=KFold(n_splits=5,random_state=self.random_state)
        #create inner and outer folds

        fold_number=0
        self.cross_validated_scores_after_rfecv=[]
        self.cross_validated_scores_after_sfscv=[]
        self.cross_validated_scores_after_perm=[]
        self.test_scores_across_all_splits=[]

        for train_index,test_index in outer_cv.split(self.X,self.y):
            
            fold_number+=1
            print('I am starting the fold %d'%fold_number)
            X_train=self.X[train_index,:]
            y_train=self.y[train_index]
            X_test=self.X[test_index,:]
            y_test=self.y[test_index]
            
            'Pipe1: REMOVE LOW VARIANCES and PERFORM STANDARD SCALER'
            print('starting pipe-1 for fold %d '%fold_number)
            pipe1=Pipeline([('lvr',running_model.Low_Variance_Remover(variance_percent=0)),('std_scaler',StandardScaler())])
            
            X_train=pipe1.fit_transform(X_train)
            X_test=pipe1.transform(X_test)
            
            scaler_y=StandardScaler()
            y_train=scaler_y.fit_transform(y_train.reshape(-1,1))
            y_test=scaler_y.transform(y_test.reshape(-1,1))

            if self.model_name=='lin_svr':
                y_train=y_train.ravel()
                y_test=y_test.ravel()
            
            combination_idx_after_pipe1=pipe1.transform(combination_idx.reshape(1,-1))
            
            save_a_model(pipe1,model_name='pipe1',split_no=fold_number,filepath=self.filepath)
            save_a_npy(combination_idx_after_pipe1,npy_name='combination_idx_after_pipe1',split_no=fold_number,filepath=self.filepath)
            #save the pipe1 and the combination indices.
            
            'Pipe-2: REMOVE HIGHLY CORRELATED FEATURES AND PERFORM SELECT_PERCENTILE [REMOVE LOW CORRELATED TO THE TARGET FEATURES]'
            print('starting pipe-2 for fold %d '%fold_number)

            pipe2=Pipeline([('hcr',running_model.High_Corr_Remover()),('select_percentile',SelectPercentile(f_regression,percentile=20))])#this is the filtering technique.
            
            pipe2.fit(X_train,y_train)
            print('I just finished with pipe2 of fold%d'%fold_number)
            
            X_train_reduced_after_pipe2=pipe2.transform(X_train)
            combination_idx_after_pipe2=pipe2.transform(combination_idx_after_pipe1)

            #we get new_combinations_after this step
            
            save_a_model(pipe2,model_name='pipe2',split_no=fold_number,filepath=self.filepath)
            save_a_npy(combination_idx_after_pipe2,npy_name='combination_idx_after_pipe2',split_no=fold_number,filepath=self.filepath)
            #save the pipe2 and the combination indices.
            
            'FINE-TUNING THE MODEL'
            
            print('I am beginning to fine tune for fold %d'%fold_number)
            fine_tuned_estimator=fine_tune_hyperparameters(param_dict=self.parameters_dict,model=self.model,X=X_train_reduced_after_pipe2,y=y_train,model_name=self.model_name,fine_tune=self.fine_tune,cv=inner_cv)
            #fine-tune hyperparameters of the regression model on the new X_train
            

            save_a_model(fine_tuned_estimator,model_name='fine_tuned_estimator',split_no=fold_number,filepath=self.filepath) #save the fine tuned esitmator
            
            '''
            CHECK FEATURE IMPORTANCES
            '''
            
            print('I am doing feature importances')
            
            indices=get_permutation_importances(fine_tuned_estimator,X=X_train_reduced_after_pipe2,y=y_train,scoring='r2')#get the indices after permutation testing
            
            combination_idx_after_perm=combination_idx_after_pipe2[:,indices]
            X_train_reduced_after_perm=X_train_reduced_after_pipe2[:,indices]
                
            save_a_npy(combination_idx_after_perm,npy_name='combination_idx_after_perm',split_no=fold_number,filepath=self.filepath) #save the new combination_indices after the second filter.
                
            scores_after_perm=cross_val_score(fine_tuned_estimator,X_train_reduced_after_perm,y_train,scoring='r2',cv=inner_cv)#get the estimated performance scores
                
            self.cross_validated_scores_after_perm.append(scores_after_perm)
                
            '''
            The cross-validated feature permutation importance is finished. Check if I want to do feature elimination (RFE vs. SFS or both)
            '''
            
            if do_feature_pruning=='none':
                print('Not doing feature elimination')
                
                fine_tuned_estimator.fit(X_train_reduced_after_perm,y_train)
                y_pred=fine_tuned_estimator.predict(X_test[:,combination_idx_after_perm.reshape(-1)])
                model_r2=r2_score(y_test,y_pred)
                self.test_scores_across_all_splits.append(model_r2)
                
                with open(self.filepath+'log.txt','a+') as file:
                    file.write('use the combination after perm for split_no %d \n'%fold_number)
                
                continue
            
            elif do_feature_pruning=='both':
                
                rfecv, rfecv_estimator, combination_idx_after_rfecv,scores_after_rfecv = performing_rfecv(fine_tuned_estimator,X_train_reduced_after_perm,y_train,step=self.step,combination_idx=combination_idx_after_perm,split_no=fold_number,filepath=self.filepath,cv=inner_cv,scoring='r2')
                
                sfscv, sfscv_estimator, combination_idx_after_sfscv,scores_after_sfscv = performing_sfscv(fine_tuned_estimator,X_train_reduced_after_perm,y_train,step=self.step,combination_idx=combination_idx_after_perm,split_no=fold_number,filepath=self.filepath,cv=inner_cv,scoring='r2')
                
                self.cross_validated_scores_after_rfecv.append(scores_after_rfecv)
                self.cross_validated_scores_after_sfscv.append(scores_after_sfscv)
            
                model_r2= get_the_best_model(X_test,y_test,self.filepath,fold_number,
                                               ('perm',fine_tuned_estimator.fit(X_train_reduced_after_perm,y_train),np.mean(scores_after_perm),combination_idx_after_perm),
                                               ('rfecv',rfecv_estimator.fit(rfecv.transform(X_train_reduced_after_perm),y_train),np.mean(scores_after_rfecv),combination_idx_after_rfecv),
                                               ('sfscv',sfscv_estimator.fit(sfscv.transform(X_train_reduced_after_perm),y_train),np.mean(scores_after_sfscv),combination_idx_after_sfscv))
                self.test_scores_across_all_splits.append(model_r2)
                
                continue
            
            elif do_feature_pruning=='rfecv':
                rfecv, rfecv_estimator, combination_idx_after_rfecv,scores_after_rfecv = performing_rfecv(fine_tuned_estimator,X_train_reduced_after_perm,y_train,step=self.step,combination_idx=combination_idx_after_perm,split_no=fold_number,filepath=self.filepath,cv=inner_cv,scoring='r2')
                
                self.cross_validated_scores_after_rfecv.append(scores_after_rfecv)
                
                model_r2= get_the_best_model(X_test,y_test,self.filepath,fold_number,
                                               ('perm',fine_tuned_estimator.fit(X_train_reduced_after_perm,y_train),np.mean(scores_after_perm),combination_idx_after_perm),
                                               ('rfecv',rfecv_estimator.fit(rfecv.transform(X_train_reduced_after_perm),y_train),np.mean(scores_after_rfecv),combination_idx_after_rfecv))
                
                self.test_scores_across_all_splits.append(model_r2)
                
                continue
            
            elif do_feature_pruning=='sfscv':
                sfscv, sfscv_estimator, combination_idx_after_sfscv,scores_after_sfscv = performing_sfscv(fine_tuned_estimator,X_train_reduced_after_perm,y_train,step=self.step,combination_idx=combination_idx_after_perm,split_no=fold_number,filepath=self.filepath,cv=inner_cv,scoring='r2')
                
                self.cross_validated_scores_after_sfscv.append(scores_after_sfscv)
                  
                model_r2= get_the_best_model(X_test,y_test,self.filepath,fold_number,
                                               ('perm',fine_tuned_estimator.fit(X_train_reduced_after_perm,y_train),np.mean(scores_after_perm),combination_idx_after_perm),
                                               ('sfscv',sfscv_estimator.fit(sfscv.transform(X_train_reduced_after_perm),y_train),np.mean(scores_after_sfscv),combination_idx_after_sfscv))
                
                self.test_scores_across_all_splits.append(model_r2)
                
                continue
            
        return self
    
if __name__ == "__main__":
    
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import LinearSVR
    from sklearn.neighbors import KNeighborsRegressor
    
    X=np.load(sys.argv[1])
    y=np.load(sys.argv[2])
    model_dict={'lin_reg':LinearRegression(),
                'lasso':Lasso(),
                'ridge':Ridge(),
                'random_forest':RandomForestRegressor(random_state=42,n_jobs=-1),
                'lin_svr':LinearSVR(),
                'knn': KNeighborsRegressor(n_jobs=-1)}
    model_name=input('model name:')
    model=model_dict[model_name]
    filepath=input('filepath:')
    # filepath='./'
    fine_tune=input('fine tune (grid/randomized):')
    x=scikit_model(model,X,y,fine_tune=fine_tune,filepath=filepath,model_name=model_name,step=1,random_state=42)
    do_feature_pruning=input('Do feature prunning?(none,both,rfecv,sfscv):')
    x.feature_selection_model(do_feature_pruning=do_feature_pruning)
    
    #saving this object for logging purposes

    import pickle
    class_name=filepath+'object'+model_name+'.pkl'
    with open(class_name,'wb') as output:
        pickle.dump(x,output,pickle.HIGHEST_PROTOCOL)
    
    
    
    
    
