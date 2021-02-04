import tqdm
import sys
import numpy as np
from codes.docs.analysis import data_preprocessing, graph_analysis

# calculating pearson correlation matrix of volumetric data

#Imperial volumetric

for i in tqdm.tqdm(range(11)):
    X_preterm=np.load('./notebooks_results/dHCP/preprocessed_data/volumetric/preterm/high_low_risk_vol_preterm_Imperial_dataset_PRS%d.npy'%i)
    y_preterm=np.load('./notebooks_results/dHCP/preprocessed_data/volumetric/preterm/high_low_risk_vol_preterm_Imperial_dataset_PRS%d_label.npy'%i)

    X_term=np.load('./notebooks_results/dHCP/preprocessed_data/volumetric/term/high_low_risk_vol_term_Imperial_dataset_PRS%d.npy'%i)
    y_term=np.load('./notebooks_results/dHCP/preprocessed_data/volumetric/term/high_low_risk_vol_term_Imperial_dataset_PRS%d_label.npy'%i)


    corrmatrix_low_risk_preterm=abs(np.corrcoef(X_preterm[y_preterm==0,:],rowvar=False))
    np.fill_diagonal(corrmatrix_low_risk_preterm,0)
    corrmatrix_high_risk_preterm=abs(np.corrcoef(X_preterm[y_preterm==1,:],rowvar=False))
    np.fill_diagonal(corrmatrix_high_risk_preterm,0)

    corrmatrix_low_risk_term=abs(np.corrcoef(X_term[y_term==0,:],rowvar=False))
    np.fill_diagonal(corrmatrix_low_risk_term,0)
    corrmatrix_high_risk_term=abs(np.corrcoef(X_term[y_term==1,:],rowvar=False))
    np.fill_diagonal(corrmatrix_high_risk_term,0)

    #generate binarized matrices for over a range of network sparsity for high and low risk groups
    binarized_low_risk_matrices_preterm=np.asarray([data_preprocessing.lower_triangle(graph_analysis.binarize_matrix_based_on_sparsity_threshold(corrmatrix_low_risk_preterm,network_sparsity_threshold/100,bins=100)) for network_sparsity_threshold in np.linspace(1,30,30)])
    binarized_high_risk_matrices_preterm=np.asarray([data_preprocessing.lower_triangle(graph_analysis.binarize_matrix_based_on_sparsity_threshold(corrmatrix_high_risk_preterm,network_sparsity_threshold/100,bins=100)) for network_sparsity_threshold in np.linspace(1,30,30)])


    binarized_low_risk_matrices_term=np.asarray([data_preprocessing.lower_triangle(graph_analysis.binarize_matrix_based_on_sparsity_threshold(corrmatrix_low_risk_term,network_sparsity_threshold/100,bins=100)) for network_sparsity_threshold in np.linspace(1,30,30)])
    binarized_high_risk_matrices_term=np.asarray([data_preprocessing.lower_triangle(graph_analysis.binarize_matrix_based_on_sparsity_threshold(corrmatrix_high_risk_term,network_sparsity_threshold/100,bins=100)) for network_sparsity_threshold in np.linspace(1,30,30)])
    
    #calculating graph_metrics_low_risk

    for threshold in tqdm.tqdm(range(30)):
        temp_path_low_risk='./notebooks_results/dHCP/log/volumetric/preterm/PRS%d/obsv_graph_metrics_vol_low_risk_preterm_Imperial_PRS%d.txt'%(i,i)
        low_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_low_risk_matrices_preterm[threshold],51))

        with open(temp_path_low_risk,'ab') as f:
            np.savetxt(f,np.asarray([low_risk_metrics]),delimiter='\t')

        temp_path_high_risk='./notebooks_results/dHCP/log/volumetric/preterm/PRS%d/obsv_graph_metrics_vol_high_risk_preterm_Imperial_PRS%d.txt'%(i,i)
        high_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_high_risk_matrices_preterm[threshold],51))

        with open(temp_path_high_risk,'ab') as n:
            np.savetxt(n,np.asarray([high_risk_metrics]),delimiter='\t')


    for threshold in tqdm.tqdm(range(30)):
        temp_path_low_risk='./notebooks_results/dHCP/log/volumetric/term/PRS%d/obsv_graph_metrics_vol_low_risk_term_Imperial_PRS%d.txt'%(i,i)
        low_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_low_risk_matrices_term[threshold],51))

        with open(temp_path_low_risk,'ab') as f:
            np.savetxt(f,np.asarray([low_risk_metrics]),delimiter='\t')

        temp_path_high_risk='./notebooks_results/dHCP/log/volumetric/term/PRS%d/obsv_graph_metrics_vol_high_risk_term_Imperial_PRS%d.txt'%(i,i)
        high_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_high_risk_matrices_term[threshold],51))

        with open(temp_path_high_risk,'ab') as n:
            np.savetxt(n,np.asarray([high_risk_metrics]),delimiter='\t')