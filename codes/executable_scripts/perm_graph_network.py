import sys
import tqdm
import numpy as np
import pandas as pd
import bct
from collections import defaultdict
from codes.docs.analysis import graph_analysis, data_preprocessing

class analyse_graph:
    def __init__(self,low_risk_corr_matrix,high_risk_corr_matrix):
        self.low_risk_corr_matrix=low_risk_corr_matrix
        self.high_risk_corr_matrix=high_risk_corr_matrix

    def network_metrics(self,sparsity_range=range(5,7)):

        for sparsity in tqdm.tqdm(sparsity_range):
            
            temp_corr_matrix_high_risk = graph_analysis.binarize_matrix_based_on_sparsity_threshold(self.high_risk_corr_matrix,sparsity/100)

            temp_corr_matrix_low_risk = graph_analysis.binarize_matrix_based_on_sparsity_threshold(self.low_risk_corr_matrix,sparsity/100)
            
            #high risk
            original_network_metrics_high, random_network_metrics_high = graph_analysis.calculate_network_metrics_bin_und(temp_corr_matrix_high_risk)

            temp_path_high_risk='./perm_graph_metrics_high_risk_vol.txt'
            with open(temp_path_high_risk,'ab') as f:
                np.savetxt(f,np.concatenate([original_network_metrics_high,random_network_metrics_high]).reshape(1,-1),delimiter='\t')

            #low risk
            original_network_metrics_low, random_network_metrics_low = graph_analysis.calculate_network_metrics_bin_und(temp_corr_matrix_low_risk)

            temp_path_low_risk='./perm_graph_metrics_low_risk_vol.txt'
            with open(temp_path_low_risk,'ab') as f:
                np.savetxt(f,np.concatenate([original_network_metrics_low,random_network_metrics_low]).reshape(1,-1),delimiter='\t')


X_total=np.load('./X_total.npy')
adjusted_prs_score=np.load('./adjusted_prs_score.npy')

for _ in tqdm.tqdm(range(1000)):
    high_risk,low_risk = data_preprocessing.high_low_risk_divide(np.random.permutation(adjusted_prs_score),high_perc=0.2,low_perc=0.4)
    X_low_risk=X_total[low_risk,:]
    X_high_risk=X_total[high_risk,:]
    high_risk_corr_matrix=np.corrcoef(X_high_risk,rowvar=False)
    low_risk_corr_matrix=np.corrcoef(X_low_risk,rowvar=False)
    high_risk_corr_matrix=0.5*(np.log(1+high_risk_corr_matrix)-np.log(1-high_risk_corr_matrix))#convert to z score
    np.fill_diagonal(high_risk_corr_matrix,0)
    low_risk_corr_matrix=0.5*(np.log(1+low_risk_corr_matrix)-np.log(1-low_risk_corr_matrix))#convert to z score
    np.fill_diagonal(low_risk_corr_matrix,0)
    
    analyse_graph(low_risk_corr_matrix,high_risk_corr_matrix).network_metrics()
    