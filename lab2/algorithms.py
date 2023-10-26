import numpy as np

from utils import get_solution_length


def two_regret(distances, start_node=None):
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

    for num_nodes in range(2, get_solution_length(len(distances))):
        biggest_regret = -np.inf
        biggest_regret_node = None
        biggest_regret_insertion = None

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
                regret = second_smallest_distance - smallest_distance
                if regret > biggest_regret:
                    biggest_regret = regret
                    biggest_regret_node = point
                    biggest_regret_insertion = current_smallest_insertion
        solution = np.insert(
            solution, biggest_regret_insertion + 1, biggest_regret_node
        )
        in_solution.add(biggest_regret_node)
    return solution


def weighted_two_regret(distances, start_node=None):
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
