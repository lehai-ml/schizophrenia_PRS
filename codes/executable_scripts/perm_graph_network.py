import tqdm
import sys
import numpy as np
from codes.docs.analysis import data_preprocessing, graph_analysis

X=np.load(sys.argv[1])

random_graph_generator=graph_analysis.yield_perm_matrices_volumetric_data(X,1000,np.linspace(1,30,30))

for _ in tqdm.tqdm(range(300)):
    try:
        binarized_perm_low_risk,binarized_perm_high_risk=next(random_graph_generator)
        
        for threshold in tqdm.tqdm(range(30)):
            temp_path_low_risk='./log/perm_graph_metrics_low_risk_vm1_run1.txt'
            perm_low_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_perm_low_risk[threshold],51),small_worldness=False)

            with open(temp_path_low_risk,'ab') as f:
                np.savetxt(f,np.asarray([perm_low_risk_metrics]),delimiter='\t')

            temp_path_high_risk='./log/perm_graph_metrics_high_risk_vm1_run1.txt'
            perm_high_risk_metrics=graph_analysis.calculate_network_metrics_random_volumetric_data(data_preprocessing.reverse_lower_triangle(binarized_perm_high_risk[threshold],51),small_worldness=False)

            with open(temp_path_high_risk,'ab') as n:
                np.savetxt(n,np.asarray([perm_high_risk_metrics]),delimiter='\t')
    except StopIteration:
        break