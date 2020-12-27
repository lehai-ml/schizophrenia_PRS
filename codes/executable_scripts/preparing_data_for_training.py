"""preparing_data_for_training.py
This script preprocesses the data and returns adjusted values of covariates and 
target data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#custom scripts
from codes.docs.analysis import data_preprocessing

print('Step 1: imports the diffusion data, structural data\n')
'''=========================================================================='''
ID,babies_connectivity_matrix=data_preprocessing.babies_connectivity_extraction('./dataset/Tracts/')

babies_structural_matrix=pd.read_csv('./dataset/ePrime-structure-volumes.csv',header=0,index_col=None).dropna()
babies_structural_matrix=babies_structural_matrix.iloc[babies_structural_matrix['Age at scan'].values<49,:].reset_index(drop=True)
'''=========================================================================='''
#importing the names for regions of interests, and their combinations.
ROIs_combinations=data_preprocessing.ROIs_combinations('./dataset/Regions_of_interests.csv')
structural_labels=pd.read_csv('./dataset/all_labels.csv', header = None, sep='\t')
structural_labels=np.asarray(structural_labels[1])

print('Step2: imports the PRS and ancestrial principle components\n')
'''=========================================================================='''
schizophrenia_high_scores_european=data_preprocessing.read_table_from_txt('./dataset/scz2_sixlevels_ld1000G.highres.all.score.txt')
ancestral_PCA_european=data_preprocessing.read_table_from_txt('./dataset/ePrime_nosibs_european_pruned_excludeLD_pca10.eigenvec.txt')
'''=========================================================================='''

print('Step3: matching europeans IDs between PRS table and diffusion and structural tables')
'''=========================================================================='''
#matching europeans IDs between tables and connectivity matrices
european_ID_list=list(set(ID).intersection(schizophrenia_high_scores_european['ID']))#this is crossing diffusion_ID with european_PRS_ID.
european_ID_list=list(set(european_ID_list).intersection(babies_structural_matrix['Subject ID']))#this is crossing european_connectivity with babies_structural_ID.
european_ID_list.sort()
european_ID_list_not_available=np.setdiff1d(schizophrenia_high_scores_european['ID'],european_ID_list)
european_ID_list_not_available.sort()
print('these %d individuals do not have connectivity matrices or structural data: %s' %(len(european_ID_list_not_available),european_ID_list_not_available))
'''=========================================================================='''

print('Step4: Removing outliers using ancestrial PCs')
'''=========================================================================='''
#outlier removal from 2 ancestral principal(?) components.

#matching available ancestrial PC ID
european_ancestral_PCA_w_outliers=data_preprocessing.extract_by_ID(european_ID_list,ancestral_PCA_european['ID'],ancestral_PCA_european)
print('the total number of subject with matching ancestrial information is %d'%european_ancestral_PCA_w_outliers.shape[0])

from scipy import stats
zscores=np.abs(stats.zscore(european_ancestral_PCA_w_outliers[:,1:3].astype('float64'),axis=0))
outliers=np.where(zscores>=2.5)[0]#remove anything more than 3 st. deviation from the mean.
print('the following IDs are outliers with zscore above 2.5 st. dev.',european_ancestral_PCA_w_outliers[outliers,0])
#removing outliers using the ancestrial PCA

fig,ax= plt.subplots()
ax.scatter(european_ancestral_PCA_w_outliers[:,1],european_ancestral_PCA_w_outliers[:,2])
for i,txt in enumerate(european_ancestral_PCA_w_outliers[outliers,0]):
    ax.annotate(txt,(european_ancestral_PCA_w_outliers[outliers,1][i],european_ancestral_PCA_w_outliers[outliers,2][i]))


#Updated lists to remove the outliers:

european_ID_list_without_outliers=data_preprocessing.extract_by_ID(european_ancestral_PCA_w_outliers[outliers,0],european_ID_list,european_ID_list,remove=True)

###########################################################

#extracting the final european connectivity matrices
european_connectivity_matrices_schizophrenia=data_preprocessing.extract_by_ID(european_ID_list_without_outliers,ID,babies_connectivity_matrix)
#extracting the final european structural matrices
european_structural_matrices_schizophrenia=data_preprocessing.extract_by_ID(european_ID_list_without_outliers,babies_structural_matrix['Subject ID'],babies_structural_matrix)

#matching available ancestrial PC ID // i am transforming all data to array list so I can work easier with scikit learn later on (is it necessary? idk)
european_ancestral_PCA=data_preprocessing.extract_by_ID(european_ID_list_without_outliers,ancestral_PCA_european['ID'],ancestral_PCA_european)

#matching available PRS score with european ID
european_prs_schizophrenia=data_preprocessing.extract_by_ID(european_ID_list_without_outliers,schizophrenia_high_scores_european['ID'],schizophrenia_high_scores_european)

'''=========================================================================='''
print('Step5: Extract ages, total volume, gender')
'''=========================================================================='''
#extracting ages, total volume, gender
european_age_at_scan=european_structural_matrices_schizophrenia[:,2]
european_gestational_age=european_structural_matrices_schizophrenia[:,4]
european_gender=european_structural_matrices_schizophrenia[:,3]
european_total_volume=european_structural_matrices_schizophrenia[:,92]
european_volumes=european_structural_matrices_schizophrenia[:,5:92]
european_icv=np.sum(european_volumes,axis=1)

#extracting the lower triangles of the connectivity matrices and make them into 1D vectors of size (4005,)
inputs_european_connectivity_matrices_schizophrenia=np.asarray(list(map(data_preprocessing.lower_triangle,european_connectivity_matrices_schizophrenia)))
connectivity_combinations=data_preprocessing.lower_triangle(ROIs_combinations)

'''=========================================================================='''
print('Step6: Remove features with zeros values')
'''=========================================================================='''
n_subjects=inputs_european_connectivity_matrices_schizophrenia.shape[0]
features_to_be_removed=np.asarray([i for i in range(0,4005) if len(np.where(inputs_european_connectivity_matrices_schizophrenia[:,i]==0)[0])/n_subjects > 0])
inputs_european_connectivity_matrices_schizophrenia_removed_zeros=np.delete(inputs_european_connectivity_matrices_schizophrenia,features_to_be_removed,axis=1)
combinations_without_zeros=np.delete(connectivity_combinations,features_to_be_removed)
print('the shape of diffusion matrix with the zeros is :',inputs_european_connectivity_matrices_schizophrenia_removed_zeros.shape)
'''=========================================================================='''
print('Step7: Adjusting for covariates')
'''=========================================================================='''
#adjusting for covariates:
#for PRS: I adjusted for 3 PCs. and for connectivity and structural data: I adjusted for age at scan, at birth, intracranial volume and gender. similar to Harriet's paper and Shi et al., 2012 (although this one has number of fibers regressed out as well).

adjusted_european_PRS_schizophrenia=data_preprocessing.adjusting_for_covariates_with_lin_reg(european_prs_schizophrenia[:,1:],european_ancestral_PCA[:,1:4])#adjust for the first 3 PCs.
adjusted_european_PRS_schizophrenia=adjusted_european_PRS_schizophrenia.astype('float64')

adjusted_european_connectivity_schizophrenia=data_preprocessing.adjusting_for_covariates_with_lin_reg(inputs_european_connectivity_matrices_schizophrenia_removed_zeros,european_age_at_scan,european_gestational_age,european_gender)

adjusted_european_structural_schizophrenia=data_preprocessing.adjusting_for_covariates_with_lin_reg(european_volumes,european_age_at_scan,european_gestational_age,european_icv,european_gender)
adjusted_european_structural_schizophrenia=adjusted_european_structural_schizophrenia.astype('float64')

#remove the CSF, Extracranial and Intracranial volume from adjusted_european_structural_schizophrenia
adjusted_european_structural_schizophrenia=np.delete(adjusted_european_structural_schizophrenia,[82,83,84],axis=1)
structural_labels=np.delete(structural_labels,[82,83,84],axis=0)

#remove the WM from adjusted_european_structural_schizophrenia retain only GM
adjusted_european_GM_structural_schizophrenia=np.delete(adjusted_european_structural_schizophrenia,[idx for idx,i in enumerate(structural_labels) if 'WM' in i],axis=1)
structural_GM_labels=np.delete(structural_labels,[idx for idx,i in enumerate(structural_labels) if 'WM' in i],axis=0)

        
    