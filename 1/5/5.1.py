import math
from random import randrange

import cv2
import numpy as np
import matplotlib.pyplot as plt

fig_w = 15
fig_h = 8


def MSE(img_x, img_y):
    return ((img_x - img_y) ** 2).sum() / (img_x.shape[0] * img_y.shape[1])


def PSNR(img_x, img_y):
    mse = MSE(img_x, img_y)
    if mse == 0:
        return math.inf
    return 10 * np.log((255 ** 2) / mse)


def random_255(img, n):
    ret = img.copy()
    for i in range(n):
        y = randrange(img.shape[0])
        x = randrange(img.shape[1])
        ret[y][x] = 255
    return ret


camera_man = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE)
print("MSE:")
print(MSE(camera_man, camera_man))
print("PSNR:")
print(PSNR(camera_man, camera_man))

noise_10 = random_255(camera_man, 100)
print(MSE(camera_man, noise_10))
print(PSNR(camera_man, noise_10))

_, subplt = plt.subplots(1, 2, figsize=(fig_w, fig_h))
subplt[0].imshow(camera_man, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(noise_10, cmap='gray')
subplt[1].set_title("Filtered Image")
plt.show()

shift_1 = camera_man + 1
print(MSE(camera_man, shift_1))
print(PSNR(camera_man, shift_1))

_, subplt = plt.subplots(1, 2, figsize=(fig_w, fig_h))
subplt[0].imshow(camera_man, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(shift_1, cmap='gray')
subplt[1].set_title("Filtered Image")
plt.show()
