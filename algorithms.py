import numpy as np
from numba import jit
from random import uniform

from utils import get_solution_length


def get_random_solution(distances, start_node=None):
    solution = np.arange(len(distances))
    np.random.shuffle(solution)
    return solution[: get_solution_length(len(distances))].astype(np.int64)


def weighted_two_regret(distances, start_node=None):
    distances = distances.copy().astype(np.float32)
    solution = np.zeros(2, dtype=np.int64)

    in_solution = set()

    if start_node is None:
        solution[0] = np.random.randint(0, len(distances))
    else:
        solution[0] = start_node
    in_solution.add(solution[0])

    solution[1] = np.argmin(distances[solution[0]])
    in_solution.add(solution[1])

    for num_nodes in range(2, get_solution_length(len(distances))):
        smallest_weighted_sum = np.inf
        smallest_weighted_sum_node = None
        smallest_weighted_sum_insertion = None

        for point in range(0, len(distances)):
            if point not in in_solution:
                smallest_distance = np.inf
                second_smallest_distance = np.inf
                current_smallest_insertion = None
                for insertion_place in range(0, num_nodes):
                    dist_change = (
                        distances[solution[insertion_place], point]
                        + distances[point, solution[(insertion_place + 1) % num_nodes]]
                        - distances[
                            solution[insertion_place],
                            solution[(insertion_place + 1) % num_nodes],
                        ]
                    )
                    if dist_change < smallest_distance:
                        second_smallest_distance = smallest_distance
                        smallest_distance = dist_change
                        current_smallest_insertion = insertion_place
                    elif dist_change < second_smallest_distance:
                        second_smallest_distance = dist_change

                # distances[point, point] is the cost of node
                # assuming node costs were added to distances matrix
                weighted_regret = distances[point, point] - (
                    second_smallest_distance - smallest_distance
                )
                if weighted_regret < smallest_weighted_sum:
                    smallest_weighted_sum = weighted_regret
                    smallest_weighted_sum_node = point
                    smallest_weighted_sum_insertion = current_smallest_insertion
        solution = np.insert(
            solution, smallest_weighted_sum_insertion + 1, smallest_weighted_sum_node
        )
        in_solution.add(smallest_weighted_sum_node)
    return solution


@jit(nopython=True)
def greedy(solution, intra_move, distances):
    best_improvement = 0
    best_solution = solution.copy()
    if uniform(0, 1) < 0.67:
        # inter moves
        inside_nodes = set(solution) 
        outside_nodes = set(range(200)) - inside_nodes
        for i, inside_node in enumerate(solution):
            before_node = solution[(i-1) %  100] 
            after_node = solution[(i+1) % 100]
            for outside_node in outside_nodes:
                improvement = distances[before_node][inside_node] \
                                + distances[inside_node][after_node] \
                                - distances[before_node][outside_node] \
                                - distances[outside_node][after_node]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_solution = solution.copy()
                    best_solution[i] = outside_node
                    return best_solution, best_improvement
    else:
        # intra moves      
        if intra_move == "node":
            for i, first_node in enumerate(solution):
                before_node_1 = solution[(i-1) %  100] 
                after_node_1 = solution[(i+1) % 100]
                for j, second_node in enumerate(solution[i+2:], i+2):
                    if j-i != 99:
                        before_node_2 = solution[(j-1) %  100] 
                        after_node_2 = solution[(j+1) % 100]
                        improvement = distances[before_node_1][first_node] \
                                        + distances[first_node][after_node_1] \
                                        + distances[before_node_2][second_node] \
                                        + distances[second_node][after_node_2] \
                                        - distances[before_node_1][second_node] \
                                        - distances[second_node][after_node_1] \
                                        - distances[before_node_2][first_node] \
                                        - distances[first_node][after_node_2]
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_solution = solution.copy()
                            best_solution[i] = second_node
                            best_solution[j] = first_node
                            return best_solution, best_improvement
                    
        elif intra_move == 'edge':
            for i, first_edge in enumerate(solution):
                before_edge_1 = solution[i] 
                after_edge_1 = solution[(i+1) % 100]
                for j, second_edge in enumerate(solution[i+2:], i+2):
                    if j-1 != 99:
                        before_edge_2 = solution[j] 
                        after_edge_2 = solution[(j+1) % 100] 
                        improvement = distances[before_edge_1][after_edge_1] \
                                        + distances[before_edge_2][after_edge_2] \
                                        - distances[before_edge_1][before_edge_2] \
                                        - distances[after_edge_1][after_edge_2] \
                                        - distances[after_edge_1][after_edge_1] \
                                        - distances[before_edge_2][before_edge_2]
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_solution = solution.copy()
                            best_solution[i+1:j+1] = best_solution[i+1:j+1][::-1]
                            return best_solution, best_improvement
    return best_solution, best_improvement


@jit(nopython=True)
def steepest(solution, intra_move, distances):
    best_improvement = 0
    best_solution = solution.copy()
    # inter moves
    inside_nodes = set(solution) 
    outside_nodes = set(range(200)) - inside_nodes
    for i, inside_node in enumerate(solution):
        before_node = solution[(i-1) %  100] 
        after_node = solution[(i+1) % 100]
        for outside_node in outside_nodes:
            improvement = distances[before_node][inside_node] \
                            + distances[inside_node][after_node] \
                            - distances[before_node][outside_node] \
                            - distances[outside_node][after_node]
            if improvement > best_improvement:
                best_improvement = improvement
                best_solution = solution.copy()
                best_solution[i] = outside_node
                    
    if intra_move == "node":
        for i, first_node in enumerate(solution):
            before_node_1 = solution[(i-1) %  100] 
            after_node_1 = solution[(i+1) % 100]
            for j, second_node in enumerate(solution[i+2:], i+2):
                if j-i != 99:
                    before_node_2 = solution[(j-1) %  100] 
                    after_node_2 = solution[(j+1) % 100]
                    improvement = distances[before_node_1][first_node] \
                                    + distances[first_node][after_node_1] \
                                    + distances[before_node_2][second_node] \
                                    + distances[second_node][after_node_2] \
                                    - distances[before_node_1][second_node] \
                                    - distances[second_node][after_node_1] \
                                    - distances[before_node_2][first_node] \
                                    - distances[first_node][after_node_2]
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_solution = solution.copy()
                        best_solution[i] = second_node
                        best_solution[j] = first_node
                
    elif intra_move == 'edge':
        for i, first_edge in enumerate(solution):
            before_edge_1 = solution[i] 
            after_edge_1 = solution[(i+1) % 100]
            for j, second_edge in enumerate(solution[i+2:], i+2):
                if j-1 != 99:
                    
                    before_edge_2 = solution[j] 
                    after_edge_2 = solution[(j+1) % 100] 
                    improvement = distances[before_edge_1][after_edge_1] \
                                    + distances[before_edge_2][after_edge_2] \
                                    - distances[before_edge_1][before_edge_2] \
                                    - distances[after_edge_1][after_edge_2] \
                                    - distances[after_edge_1][after_edge_1] \
                                    - distances[before_edge_2][before_edge_2]
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_solution = solution.copy()
                        best_solution[i+1:j+1] = best_solution[i+1:j+1][::-1]
    return best_solution, best_improvement