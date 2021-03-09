import numpy as np
import pandas as pd
import tqdm
import statsmodels.api as sm
import pickle

PRS_thresholds=['X1e.08','X1e.07','X1e.06','X1e.05','X0.0001','X0.001','X0.01','X0.05', 'X0.1', 'X0.5', 'X1']
for i in tqdm.tqdm(range(500)):
    new_diffusion_dataset=pd.read_csv('./reduced_european_diffusion_dataset.csv')
    permuted_prs_values=np.random.permutation(new_diffusion_dataset.loc[:,PRS_thresholds])
    new_diffusion_dataset.loc[:,PRS_thresholds]=permuted_prs_values
    model_list={}
    filename='./log/model_perm_'+str(i)+'.pkl'
    for threshold in tqdm.tqdm(PRS_thresholds):
        X=new_diffusion_dataset[['GA.at.birth','PMA.at.birth','Gender','Anc_PC1','Anc_PC2','Anc_PC3',threshold]].copy()
        X=sm.add_constant(X)
        model_list[threshold]={}
        for connection in new_diffusion_dataset.columns[5:2518]:
            y=new_diffusion_dataset[connection]
            model=sm.OLS(y,X).fit()
            model_list[threshold][connection]=list(model.pvalues)
            model_list[threshold][connection].append(threshold)
    with open(filename, 'wb') as fp:
        pickle.dump(model_list,fp,protocol=pickle.HIGHEST_PROTOCOL)
