import matplotlib.pyplot as plt
import numpy as np


def print_stats(scores, name_alg):
    print(f"Min score for {name_alg}: {min(scores)}")
    print(f"Max score for {name_alg}: {max(scores)}")
    print(f"Avg score for {name_alg}: {np.mean(scores)}")


def visualize_solution(data, solution):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    size = data.iloc[:, 2] / 10

    plt.scatter(x, y, s=size, c="b", label="All Points")

    for i in range(len(solution) - 1):
        point1 = solution[i]
        point2 = solution[i + 1]
        plt.plot([x[point1], x[point2]], [y[point1], y[point2]], "r")

    point1 = solution[-1]
    point2 = solution[0]
    plt.plot([x[point1], x[point2]], [y[point1], y[point2]], "r")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Solution Visualization")
    plt.show()


def visualize_stats(scores, name_alg):
    print_stats(scores, name_alg)
    plt.hist(scores, alpha=0.5, label="random", color="green")
    plt.vlines(min(scores), 0, 200, color="green")
    plt.vlines(max(scores), 0, 200, color="green")
    plt.vlines(np.mean(scores), 0, 200, color="green")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title(f"Histogram for {name_alg}")
    plt.show()
