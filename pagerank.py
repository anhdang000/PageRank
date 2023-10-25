import time

import numpy as np
from scipy.sparse import csc_matrix


def parse_graph(file_path):
    # Read input edges
    with open(file_path) as f:
        edges = [line.strip().split(',') for line in f]

    # Find maximum node id    
    max_node_id = max([max(map(int, edge)) for edge in edges])

    # Create adjacency matrix
    adjacency_matrix = np.zeros((max_node_id, max_node_id), dtype='bool')
    for edge in edges:
        node_a, node_b = map(int, edge)
        adjacency_matrix[node_a - 1][node_b - 1] = True
    return adjacency_matrix



def page_rank(G, d=0.85, n_iter=100, tol=1e-6):
    """
    Compute the PageRank scores for a given graph.

    Parameters:
        - G: numpy array, adjacency matrix of the graph
        - d: float, damping factor (default: 0.85)
        - n_iter: int, maximum number of iterations (default: 100)
        - tol: float, tolerance for convergence (default: 1e-6)

    Returns:
        - final_state: numpy array, PageRank scores for each node in the graph
    """

    n_nodes = G.shape[0]
    
    # Transition  matrix
    transition_mat = G / (np.sum(G, axis=1).reshape(-1,1) + 1e-6)

    # Random surfer matrix
    random_surf = np.ones(shape=(n_nodes, n_nodes)) / n_nodes    

    # Iterate over G rows to find dangling nodes (where that row contains all zeros)
    dangling_nodes = np.zeros(shape = (n_nodes,))
    for i in range(n_nodes):
        if np.sum(G[i, :]) == 0:
            dangling_nodes[i] = 1
    dangling_node_mat = np.repeat(dangling_nodes[:,np.newaxis], n_nodes, axis=1) / n_nodes
    stochastic_mat = transition_mat + dangling_node_mat
    
    # Compute pagerank matrix
    pagerank_mat = d * stochastic_mat + (1-d) * random_surf

    state_0 = np.ones(shape=(n_nodes,)) / n_nodes
    state = state_0
    for i in range(n_iter):
        final_state = np.transpose(pagerank_mat) @ state

        state_0 = state
        state = final_state

        err = np.mean(np.abs(state - state_0))
        if err < tol:
            print(f'Done in {i} iterations')
            break
    return final_state


if __name__=='__main__':
    G = parse_graph('data/graph1.txt')
    G = np.array(
        [
            [0,0,1,0,0,0,0],
            [0,0,1,0,1,0,0],
            [1,0,0,1,0,0,1],
            [1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1],
            [1,0,1,0,0,0,1],
            [0,0,0,1,1,0,0]
        ]
    )
    print(page_rank(G, d=0.85))
