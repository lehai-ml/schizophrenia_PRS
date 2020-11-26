#derive the score

#calculate_the_scores from the graphs objects
import time
from model_training import *
from networkx.algorithms.efficiency_measures import global_efficiency,local_efficiency
import sys


if __name__ == "__main__":
    
    starttime=time.time()

    random_low_risk_graphs=load_the_object('./randomized_graph/random_low_risk_graphs.pkl')
    random_high_risk_graphs=load_the_object('./randomized_graph/random_high_risk_graphs.pkl')
    start=int(input('start'))
    end=int(input('end'))
    random_low_risk_graph_metrics={'global_efficiency':{},'local_efficiency':{}}
    random_high_risk_graph_metrics={'global_efficiency':{},'local_efficiency':{}}

    for i in range(start,end):
        random_low_risk_graph_metrics['global_efficiency'][i]=[global_efficiency(n) for n in random_low_risk_graphs[i]]
        random_high_risk_graph_metrics['global_efficiency'][i]=[global_efficiency(n) for n in random_high_risk_graphs[i]]

        random_low_risk_graph_metrics['local_efficiency'][i]=[local_efficiency(n) for n in random_low_risk_graphs[i]]
        random_high_risk_graph_metrics['local_efficiency'][i]=[local_efficiency(n) for n in random_high_risk_graphs[i]]

    print(time.time()-starttime)
    save_the_object(random_low_risk_graph_metrics,'random_low_risk_graph_metrics'+str(start)+str(end)+'.pkl')
    save_the_object(random_high_risk_graph_metrics,'random_high_risk_graph_metrics'+str(start)+str(end)+'.pkl')