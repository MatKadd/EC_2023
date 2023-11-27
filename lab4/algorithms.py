import numpy as np


from random import uniform
from numba import jit
from numba.typed import List


@jit(nopython=True)
def get_random_solution(start_node=None):
    solution = np.arange(200)
    np.random.shuffle(solution)
    return solution[:100].astype(np.int64)


@jit(nopython=True)
def get_edge_neigh(solution, candidate_edges):
    neigh = List()
    inside_nodes = set(solution) 
    outside_nodes = set(range(200)) - inside_nodes
    for i, inside_node in enumerate(solution):
        for outside_node in outside_nodes:
            before_node = solution[(i-1) %  100] 
            after_node = solution[(i+1) % 100]
            if (outside_node in candidate_edges[before_node]) or (after_node in candidate_edges[outside_node]):
                neigh.append((0, i, outside_node))

    for i, first_edge in enumerate(solution):
        for j, second_edge in enumerate(solution[i+2:], i+2):
            if j-i != 99:
                before_edge_1 = solution[i] 
                after_edge_1 = solution[(i+1) % 100]
                before_edge_2 = solution[j] 
                after_edge_2 = solution[(j+1) % 100] 
                if (before_edge_2 in candidate_edges[before_edge_1]) or (after_edge_2 in candidate_edges[after_edge_1]):
                    neigh.append((1, i, j))
    return neigh


@jit(nopython=True)
def steepest(solution, neigh, distances):
    best_solution = solution.copy()
    best_improvement = 0
    for move in neigh:
        if move[0] == 0:
            i = move[1]
            inside_node = solution[i]
            before_node = solution[(i-1) %  100] 
            after_node = solution[(i+1) % 100]
            outside_node = move[2]
            improvement = distances[before_node][inside_node] \
                            + distances[inside_node][after_node] \
                            - distances[before_node][outside_node] \
                            - distances[outside_node][after_node]
            if improvement > best_improvement:
                best_improvement = improvement
                best_solution = solution.copy()
                best_solution[i] = outside_node
                
        elif move[0] == 1:
            i = move[1]
            j = move[2]
            before_edge_1 = solution[i] 
            after_edge_1 = solution[(i+1) % 100]
            before_edge_2 = solution[j] 
            after_edge_2 = solution[(j+1) % 100] 
            improvement = distances[before_edge_1][after_edge_1] \
                            + distances[before_edge_2][after_edge_2] \
                            - distances[before_edge_1][before_edge_2] \
                            - distances[after_edge_1][after_edge_2] \
                            - distances[after_edge_1][after_edge_1] \
                            + distances[before_edge_2][before_edge_2]
            if improvement > best_improvement:
                best_improvement = improvement
                best_solution = solution.copy()
                best_solution[i+1:j+1] = best_solution[i+1:j+1][::-1]
    return best_solution, best_improvement