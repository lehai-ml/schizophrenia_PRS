import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def nx_kamada_kawai_layout(test_graph):
    '''
    Input: requires the networkx graph object
    '''
    weights=nx.get_edge_attributes(test_graph,'weight').values()
    pos=nx.kamada_kawai_layout(test_graph)
    node_hubs=[(node,degree) for node,degree in sorted(dict(test_graph.degree).items(),key=lambda item:item[1],reverse=True)][:5] #sort dictionary by the values in the descending order
    node_hubs_names=[node for node,degree in node_hubs]
    labels={}
    for node in test_graph.nodes:
        if node in node_hubs_names:
            #set the node name as the key and the label as its value 
            labels[node] = node
    #set the argument 'with labels' to False so you have unlabeled graph
    nx.draw(test_graph,pos,width=list(weights),node_size=50,node_color='lightgreen',with_labels=False)
    #Now only add labels to the nodes you require
    nx.draw_networkx_labels(test_graph,pos,labels,font_size=16,font_color='r')


def printProgressBar (iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
if __name__ == "__main__":
    pass