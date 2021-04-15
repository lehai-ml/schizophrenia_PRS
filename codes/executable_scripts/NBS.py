import tqdm
import sys
import numpy as np
from codes.docs.analysis import data_preprocessing, MLtraining

european_diffusion_dataset_term=pd.read_csv('./european_diffusion_dataset_term.csv')

X_total=np.asarray(european_diffusion_dataset_term.loc[:,'PreCG.R_PreCG.L':'ITG.R_ITG.L'])
feature_column_names=european_diffusion_dataset_term.loc[:,'PreCG.R_PreCG.L':'ITG.R_ITG.L'].columns

retain_non_zero=MLtraining.Retain_non_zero_features().fit(X_total)
X_total[:,retain_non_zero.zero_perc>0]=np.nan # keep the 4005 features, but set the  sparse ones to np.nan

covariates=np.asarray(european_diffusion_dataset_term[['GA_diff','PMA_diff','Gender','Anc_PC1','Anc_PC2','Anc_PC3']])
target=np.asarray(european_diffusion_dataset_term['PRS_0.05'])


#permute the prs score
permuted_target=(np.random.permutation(target) for _ in range(10))

for _ in tqdm.tqdm(range(2)):
    permuted_data=MLtraining.Select_Features_Multivariate(target=next(permuted_target),covariates=covariates).fit(X_total)
    min_p_value=np.min(permuted_data.p_value[~np.isnan(permuted_data.p_value)])
    largest_component_size=permuted_data.return_largest_component_size()
    with open('./result.txt','a') as file:
        np.savetxt(file,np.asarray([min_p_value,largest_component_size]).reshape(1,-1),delimiter='\t',fmt="%f")
