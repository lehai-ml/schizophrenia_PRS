from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import MLtraining.model_training_simple as model_training_simple
import numpy as np
import sys
X=np.load(sys.argv[1])
y=np.load(sys.argv[2])
combination_idx=np.load(sys.argv[3])
model_dict={'lin_reg':LinearRegression(),
            'lasso':Lasso(),
            'ridge':Ridge(),
            'random_forest':RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1),
            'lin_svr':LinearSVR(),
            'knn': KNeighborsRegressor(n_jobs=-1)}
model_name=input('model name:')
model=model_dict[model_name]
filepath=input('filepath:')
# filepath='./'
fine_tune=input('fine tune (grid/randomized):')
x=model_training_simple.scikit_model(model,X,y,combination_idx=combination_idx,fine_tune=fine_tune,filepath=filepath,model_name=model_name,random_state=42)
do_feature_pruning=input('Do feature prunning?(none,both,rfecv,sfscv):')
x.feature_selection_model_simple(do_feature_pruning=do_feature_pruning)

#saving this object for logging purposes

import pickle
class_name=filepath+'object'+model_name+'.pkl'
with open(class_name,'wb') as output:
    pickle.dump(x,output,pickle.HIGHEST_PROTOCOL)
