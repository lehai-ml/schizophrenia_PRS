import tqdm
import sys
import numpy as np
from codes.docs.analysis import data_preprocessing, graph_analysis

X1=np.load(sys.argv[1])
X2=np.load(sys.argv[2])

random_graph_generator_preterm=graph_analysis.yield_perm_matrices_volumetric_data(X1,1000,np.linspace(1,30,30),diffusion_data=True)

random_graph_generator_term=graph_analysis.yield_perm_matrices_volumetric_data(X2,1000,np.linspace(1,30,30),diffusion_data=True)

# for _ in tqdm.tqdm(range(1000)):
#     try:
#         binarized_perm_low_risk,binarized_perm_high_risk=next(random_graph_generator_preterm)
        
#         for threshold in tqdm.tqdm(range(30)):
#             temp_path_low_risk='./log/diffusion/preterm/PRS10/perm_graph_metrics_low_risk_diffusion_preterm_PRS10_vm1_run1.txt'
#             perm_low_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_perm_low_risk[threshold],51),n_random=10,small_worldness=True)

#             with open(temp_path_low_risk,'ab') as f:
#                 np.savetxt(f,np.asarray([perm_low_risk_metrics]),delimiter='\t')

#             temp_path_high_risk='./log/diffusion/preterm/PRS10/perm_graph_metrics_high_risk_diffusion_preterm_PRS10_vm1_run1.txt'
#             perm_high_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_perm_high_risk[threshold],51),n_random=10,small_worldness=True)

#             with open(temp_path_high_risk,'ab') as n:
#                 np.savetxt(n,np.asarray([perm_high_risk_metrics]),delimiter='\t')
#     except StopIteration:
#         break
    
print('STARTING TERM')

for _ in tqdm.tqdm(range(500)):
    try:
        binarized_perm_low_risk,binarized_perm_high_risk=next(random_graph_generator_term)
        
        for threshold in tqdm.tqdm(range(30)):
            temp_path_low_risk='./log/diffusion/term/PRS8/perm_graph_metrics_low_risk_diffusion_term_PRS8_vm2_run1.txt'
            perm_low_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_perm_low_risk[threshold],51),n_random=10,small_worldness=True)

            with open(temp_path_low_risk,'ab') as f:
                np.savetxt(f,np.asarray([perm_low_risk_metrics]),delimiter='\t')

            temp_path_high_risk='./log/diffusion/term/PRS8/perm_graph_metrics_high_risk_diffusion_term_PRS8_vm2_run1.txt'
            perm_high_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_perm_high_risk[threshold],51),n_random=10,small_worldness=True)

            with open(temp_path_high_risk,'ab') as n:
                np.savetxt(n,np.asarray([perm_high_risk_metrics]),delimiter='\t')
    except StopIteration:
        break