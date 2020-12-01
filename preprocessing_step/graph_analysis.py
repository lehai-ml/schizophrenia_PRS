#graph_analysis.py
import preprocessing_step.preprocessing as preprocessing
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.cluster import clustering
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from networkx.exception import NetworkXError

from more_itertools import random_combination

def create_a_graph(Graph,edge_list,weight_list=None):
    source_list=[]
    target_list=[]
    for edge in edge_list:
        source_list.append(edge[0])
        target_list.append(edge[1])
    edges_df=pd.DataFrame(
        {
            "source":source_list,
            "target":target_list,
            "weight":weight_list
        }
    )
    Graph=nx.from_pandas_edgelist(edges_df,edge_attr=True)
    return Graph

def binarize_matrix_based_on_sparsity_threshold(corrmatrix,percentage,bins=10):
    connections=preprocessing.lower_triangle(corrmatrix,corrmatrix.shape[0])
    counts,the_bin=np.histogram(connections,bins=bins)
    number_to_retain=percentage*len(connections)
    cut_off_idx=np.where(np.cumsum(counts[::-1])>=number_to_retain)[0][0]
    threshold=the_bin[::-1][cut_off_idx+1]
    binarized_corrmatrix=np.where(corrmatrix[:,:]>threshold,float(1),float(0))
    return binarized_corrmatrix

def create_random_graph(original_graph,r):
    """
    Generate randomized graph:In the original matrix, if i1 was connected to j1 and i2 was connected to j2, for random matrices, we swapped the edges between i1 and i2.(Shi et al., 2012)

    """
    for repetition in range(r):
        updated_graph=original_graph.copy()
        random_graph=nx.Graph()
        number_of_edges=len(updated_graph.edges)
        for edge in range(number_of_edges):
            try:
                i,j=random_combination(updated_graph.edges,2)
                rewired_pair=((i[0],j[1]),(j[0],i[1]))
                updated_graph.remove_edges_from((i,j))
                random_graph.add_edges_from(rewired_pair)
            except ValueError:
                if len(updated_graph.edges)==1:
                    random_graph.add_edges_from(updated_graph.edges)
        yield random_graph
        
        
def clustering_coefficient_and_path_length(G):
    clustering_values=clustering(G)
    clustering_values=np.mean([abs(value) for value in clustering_values.values()])
    try:
        path_length=average_shortest_path_length(G)
    except NetworkXError:
        path_length=[]
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            path_length.append(average_shortest_path_length(C))
        path_length=np.mean(path_length)
    return clustering_values,path_length