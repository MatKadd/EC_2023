import numpy as np


def calculate_distances(instance):
    distances = np.zeros((len(instance), len(instance)))
    for i in range(len(instance)):
        for j in range(len(instance)):
            distances[i, j] = np.sqrt(
                (instance.iloc[i, 0] - instance.iloc[j, 0]) ** 2
                + (instance.iloc[i, 1] - instance.iloc[j, 1]) ** 2
            )

    distances = np.round(distances).astype(np.int32)
    distances += np.array(instance.iloc[:, 2])
    return distances


def get_solution_length(num_of_nodes):
    return num_of_nodes // 2 + num_of_nodes % 2


def evaluate_solution(solution, distances):
    return np.sum(distances[solution, np.roll(solution, -1)])


def get_scores(algorithm, distances):
    solutions = [algorithm(distances, start_node=_) for _ in range(200)]
    scores = [evaluate_solution(solution, distances) for solution in solutions]
    best_sol = solutions[np.argmin(scores)]
    return scores, best_sol
