"""
Executing the finetuning and model selection. This is used for regression 
    model, but can be modified to fit other problems.
"""


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel,SelectPercentile,f_regression,RFECV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,RandomizedSearchCV, cross_validate
from sklearn.metrics import mean_squared_error

from sklearn.externals import joblib

import inspect
import running_model

import numpy as np


    
    
    


def save_a_model(model,model_name,split_no,filepath):
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
    filepath=filepath+'split_'+str(split_no)+model_name+'.pkl'
    return joblib.dump(model,filepath)

def save_a_npy(npy_array,npy_name,split_no,filepath):
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
    filepath=filepath+'split_'+str(split_no)+npy_name+'.npy'
    return np.save(filepath,npy_array)


def fine_tune_hyperparameters(param_dict,model,X,y,model_name,fine_tune='grid',cv=4,scoring='neg_mean_squared_error'):
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
            print('There is no parameters to fine_tune, if this is not the case, please add parameters to the model_training.parameters_dict if that is not correct')
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
            search=RandomizedSearchCV(model,param_distributions=param_grid,cv=cv,scoring=scoring)
        
        search.fit(X,y)
        
        examined_param_grid[model_examined]['search_model_best_estimator']=search.best_estimator_# save into that dictionary the best_estimator, best_parameter and best scores.
        examined_param_grid[model_examined]['search_best_params']=search.best_params_
        examined_param_grid[model_examined]['best_score']=np.sqrt(-search.best_score_)

        cvres=search.cv_results_
        
        print('here is the fine tuning results \n')
        for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
            print(np.sqrt(-mean_score),params)
        
        loop_continue=input('Do you want to repeat? (yes/no)')
        if loop_continue=='no':
            print('which model you want to choose:')# choose from the list the best model
            n=[print(i) for i in [(key,value['search_best_params'],value['best_score']) for key,value in examined_param_grid.items()]]
            the_model_i_want=eval(input('select a model number'))
            return examined_param_grid[the_model_i_want]['search_model_best_estimator']
            break

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
                         'random_forest':{'n_estimators':[3,10,30],
                         'max_features':[2,4,6,8]},
                         'svm':{},
                         'knn':{}})

    def feature_selection_model(self,combination_idx=np.arange(4005)):
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
            SelectFromModel to select for the highest "feature_importance" or 
            "coef_" features, as defined by the model.
            
        Fifth, the model is then passed to the RFECV, to prune the rest of the 
            features.
        
        Args:
            combination_idx= the features idx.
        
        Returns:
            saves externally the combination_idx 
            self.cross_validated_scores_across_all_splits
            self.test_scores_across_all_split
        """
        inner_cv=KFold(n_splits=4,random_state=self.random_state)
        outer_cv=KFold(n_splits=5,random_state=self.random_state)
        #create inner and outer folds
        
        fold_number=0
        self.cross_validated_scores_across_all_splits=[]
        self.test_scores_across_all_splits=[]

        for train_index,test_index in outer_cv.split(self.X,self.y):
            
            fold_number+=1
            
            X_train=self.X[train_index,:]
            y_train=self.y[train_index]
            X_test=self.X[test_index,:]
            y_test=self.y[test_index]
            
            scaler_X=StandardScaler()
            X_train=scaler_X.fit_transform(X_train)
            X_test=scaler_X.transform(X_test)
            
            scaler_y=StandardScaler()
            y_train=scaler_y.fit_transform(y_train.reshape(-1,1))
            y_test=scaler_y.transform(y_test.reshape(-1,1))
            #Scaling the data
            
            pipe1=Pipeline([('featureRed',running_model.FeatureReduction()),('select_percentile',SelectPercentile(f_regression,percentile=20))])#this part is the filtering technique.
            
            pipe1.fit(X_train,y_train)
            
            X_train_reduced_after_pipe1=pipe1.transform(X_train)
            combination_idx_after_pipe1=pipe1.transform(combination_idx.reshape(1,-1))

            #we get new_combinations_after this step
            
            save_a_model(pipe1,model_name='pipe1',split_no=fold_number,filepath=self.filepath)
            save_a_npy(combination_idx_after_pipe1,npy_name='combination_idx_after_pipe1',split_no=fold_number,filepath=self.filepath)
            #save the pipe1 and the combination indices.
            
            fine_tuned_estimator=fine_tune_hyperparameters(param_dict=self.parameters_dict,model=self.model,X=X_train_reduced_after_pipe1,y=y_train,model_name=self.model_name,fine_tune=self.fine_tune,cv=inner_cv)
            #fine-tune hyperparameters of the regression model on the new X_train
            
            
            save_a_model(fine_tuned_estimator,model_name='fine_tuned_estimator',split_no=fold_number,filepath=self.filepath) #save the fine tuned esitmator
            
            sfm=SelectFromModel(fine_tuned_estimator,prefit=True)
            
            X_train_reduced_after_sfm2=sfm.transform(X_train_reduced_after_pipe1)
            combination_idx_after_sfm2=sfm.transform(combination_idx_after_pipe1)
            save_a_npy(combination_idx_after_sfm2,npy_name='combination_idx_after_sfm2',split_no=fold_number,filepath=self.filepath)
            #save the new combination_indices after the second filter.
            
            rfecv=RFECV(estimator=sfm.estimator,step=self.step,scoring='neg_mean_squared_error').fit(X_train_reduced_after_sfm2,y_train)
            
            combination_idx_after_rfecv=rfecv.transform(combination_idx_after_sfm2)
            
            save_a_npy(combination_idx_after_rfecv,npy_name='combination_idx_after_rfecv',split_no=fold_number,filepath=self.filepath)
            #save the new combination indices after the final filter.
            
            save_a_model(rfecv,model_name='rfecv',split_no=fold_number,filepath=self.filepath)
            #save the rfecv model
        
            rfecv.estimator.fit(rfecv.transform(X_train_reduced_after_sfm2),y_train)
            scores=cross_val_score(rfecv.estimator,rfecv.transform(X_train_reduced_after_sfm2),y_train,scoring='neg_mean_squared_error',cv=inner_cv) #get the estimated performance scores
            
            self.cross_validated_scores_across_all_splits.append(scores)
            
            y_pred=rfecv.estimator.predict(X_test[:,combination_idx_after_rfecv.reshape(-1)])
            
            model_rmse=np.sqrt(mean_squared_error(y_test,y_pred))
            self.test_scores_across_all_splits.append(model_rmse)
            
        return self
        
        
if __name__ == "__main__":
    pass
    