import numpy as np

from utils import get_solution_length


def get_random_solution(distances, start_node=None):
    solution = np.arange(len(distances))
    np.random.shuffle(solution)
    return solution[: get_solution_length(len(distances))]


def get_nearest_neighbor_solution(distances, start_node=None):
    distances = distances.copy().astype(np.float32)
    solution = np.zeros(get_solution_length(len(distances)), dtype=np.int32)
    if start_node is None:
        solution[0] = np.random.randint(0, len(distances))
    else:
        solution[0] = start_node
    distances[:, solution[0]] = np.inf

    for i in range(1, len(solution)):
        solution[i] = np.argmin(distances[solution[i - 1]])
        distances[:, solution[i]] = np.inf
    return solution


def get_greedy_cycle_solution(distances, start_node=None):
    distances = distances.copy().astype(np.float32)
    solution = np.zeros(2, dtype=np.int32)

    in_solution = set()

    if start_node is None:
        solution[0] = np.random.randint(0, len(distances))
    else:
        solution[0] = start_node
    in_solution.add(solution[0])

    solution[1] = np.argmin(distances[solution[0]])
    in_solution.add(solution[1])

    num_nodes = 2
    for i in range(2, get_solution_length(len(distances))):
        min_increase = np.inf
        best_node = None
        edge_start = None
        for j in range(0, num_nodes - 1):
            for k in range(0, len(distances)):
                if k not in in_solution:
                    dist_change = (
                        distances[solution[j], k]
                        + distances[k, solution[j + 1]]
                        - distances[solution[j], solution[j + 1]]
                    )
                    if dist_change < min_increase:
                        min_increase = dist_change
                        best_node = k
                        edge_start = j

        for k in range(0, len(distances)):
            if k not in in_solution:
                dist_change = (
                    distances[solution[num_nodes - 1], k]
                    + distances[k, solution[0]]
                    - distances[solution[num_nodes - 1], solution[0]]
                )
                if dist_change < min_increase:
                    min_increase = dist_change
                    best_node = k
                    edge_start = num_nodes - 1

        solution = np.insert(solution, edge_start + 1, best_node)
        in_solution.add(best_node)
        num_nodes += 1
    return solution
