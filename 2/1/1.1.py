import numpy as np
from matplotlib import pyplot as plt

n = 8
base_pic = np.array(
    [[[[np.exp(-2j * np.pi * (i * u + j * v) / n) for j in range(n)] for i in range(n)] for v in range(n)]
     for u in range(n)])

_, subplt = plt.subplots(n, n, figsize=(n, n))
for u in range(n):
    for v in range(n):
        subplt[u][v].imshow(base_pic[u][v].real, vmin=-1, vmax=1, cmap='gray')
        subplt[u][v].set_axis_off()
plt.show()

_, subplt = plt.subplots(n, n, figsize=(n, n))
for u in range(n):
    for v in range(n):
        subplt[u][v].imshow(base_pic[u][v].imag, vmin=-1, vmax=1, cmap='gray')
        subplt[u][v].set_axis_off()
plt.show()
