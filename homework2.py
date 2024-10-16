# завдання 1

import numpy as np


def vector_lengths(vectors):
    squared_norms = np.sum(vectors ** 2, axis=1)
    return np.sqrt(squared_norms)


vectors = np.array([[9, 3], [7, 2], [8, 4]])
lengths = vector_lengths(vectors)
print(lengths)


# завдання 2


def vector_angles(x, y):
    dot_products = np.sum(x * y, axis=1)
    norms_X = np.sqrt(np.sum(x ** 2, axis=1))
    norms_Y = np.sqrt(np.sum(y ** 2, axis=1))
    cosines = dot_products / (norms_X * norms_Y)
    cosines = np.clip(cosines, -1, 1)
    radians = np.arccos(cosines)
    return np.degrees(radians)


X = np.array([[10, 3], [5, 7], [0, 1]])
Y = np.array([[6, 1], [-1, 2], [8, 9]])

angles = vector_angles(X, Y)

print(angles)


# завдання 3

def meeting_lines(a1, b1, a2, b2):
    a = np.array([[1, -a1], [1, -a2]])
    b = np.array([b1, b2])

    x, y = np.linalg.solve(a, b)
    return x, y


a1 = 7
b1 = 8
a2 = 3
b2 = 1

x, y = meeting_lines(a1, b1, a2, b2)

print("точка зустрічі двох ліній:", x, y)

# завдання 4

from functools import reduce


def matrix_power(a, n):
    if n == 0:
        return np.eye(a.shape[0])
    elif n > 0:
        return reduce(np.matmul, [a] * n)
    else:
        return reduce(np.matmul, [np.linalg.inv(a)] * -n)


A = np.array([[7, 4],
              [1, 8]])

print(matrix_power(A, -4))