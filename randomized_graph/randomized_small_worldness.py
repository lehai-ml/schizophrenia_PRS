import numpy as np
import MLtraining
import preprocessing_step as preprocessing

random_low_risk_graphs=MLtraining.load_the_object('random_low_risk_graphs.pkl')
# random_high_risk_graphs=MLtraining.load_the_object('./randomized_graph/permutation_statistics/random_high_risk_graphs.pkl')

##low_risk##

C_sparsity_normalized_random_low_risk=[]
L_sparsity_normalized_random_low_risk=[]

for sparsity in range(30):
    C_temp=[]
    L_temp=[]
    for perm in range(100):
        x=preprocessing.create_random_graph(random_low_risk_graphs[perm][sparsity],10)
        
        C_rand=[]
        L_rand=[]
        for m in range(10):
            clustering_values,path_length=preprocessing.clustering_coefficient_and_path_length(next(x))
            C_rand.append(clustering_values)
            L_rand.append(path_length)
        
        C_real,L_real=preprocessing.clustering_coefficient_and_path_length(random_low_risk_graphs[perm][sparsity])
        
        C_normalized_temp=C_real/np.mean(C_rand)
        L_normalized_temp=L_real/np.mean(L_rand)
        
        C_temp.append(C_normalized_temp)
        L_temp.append(L_normalized_temp)
    
    C_sparsity_normalized_random_low_risk.append(C_temp)
    L_sparsity_normalized_random_low_risk.append(L_temp)


save_the_object(C_sparsity_normalized_random_low_risk,'C_sparsity_normalized_random_low_risk.pkl')
save_the_object(L_sparsity_normalized_random_low_risk,'L_sparsity_normalized_random_low_risk.pkl')

    
        
        
    
    