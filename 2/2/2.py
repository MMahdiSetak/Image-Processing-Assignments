import numpy as np

n = 4
T = np.array([[np.exp(-2j * np.pi * i * j / n) for j in range(n)] for i in range(n)]) / n

a = np.array([1, 0, 1, 0])
print(np.round(a.dot(T), 5))

b = np.array([2, -2, 1j, 0])
print(np.round(b.dot(T)))

n = 5
T = np.array([[np.exp(-2j * np.pi * i * j / n) for j in range(n)] for i in range(n)]) / n

p = np.array([1, 1, 1, 1, 1])
print(np.round(p.dot(T)))

t = np.array([1j, 1j, 1j, 1j, 1j])
print(np.round(t.dot(T)))
