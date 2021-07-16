import cv2
import numpy as np
from skimage import img_as_float
from skimage.util import random_noise
from skimage.morphology import disk
from skimage.filters import median
import matplotlib.pyplot as plt

# camera_man = img_as_float(cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE))
camera_man = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE)
babon = cv2.imread("mandril.tiff", cv2.IMREAD_GRAYSCALE)
# babon = cv2.cvtColor(babon, cv2.COLOR_BGR2RGB)

salty_camera_man = random_noise(camera_man, mode='s&p')
salty_babon = random_noise(babon, mode='s&p', amount=0.1)

sigma = 0.1
gaussian_camera_man = random_noise(camera_man, mode='gaussian', var=sigma ** 2, mean=0)
gaussian_babon = random_noise(babon, mode='gaussian', var=sigma ** 2, mean=0)

f, subplt = plt.subplots(2, 3, figsize=(15, 10))
subplt[0][0].imshow(camera_man, cmap='gray')
subplt[0][0].set_title("Original Cameraman")
subplt[0][1].imshow(salty_camera_man, cmap='gray')
subplt[0][1].set_title("Salty Cameraman")
subplt[0][2].imshow(gaussian_camera_man, cmap='gray')
subplt[0][2].set_title("Gaussian Cameraman")
subplt[1][0].imshow(babon, cmap='gray')
subplt[1][0].set_title("Original Babon")
subplt[1][1].imshow(salty_babon, cmap='gray')
subplt[1][1].set_title("Salty Babon")
subplt[1][2].imshow(gaussian_babon, cmap='gray')
subplt[1][2].set_title("Gaussian Babon")
plt.show()

gate3x3 = [[1, 2, 1],
           [2, 4, 2],
           [1, 2, 1]]
g_kernel3x3 = np.array(gate3x3, dtype=np.float32) / 16
avg_kernel3x3 = np.ones((3, 3), dtype=np.float32) / 9
gate5x5 = [[1, 1, 2, 1, 1],
           [1, 2, 4, 2, 1],
           [2, 4, 8, 4, 2],
           [1, 2, 4, 2, 1],
           [1, 1, 2, 1, 1]]
g_kernel5x5 = np.array(gate5x5, dtype=np.float32) / 52
avg_kernel5x5 = np.ones((5, 5), dtype=np.float32) / 25

camera_man_gaussian3 = cv2.filter2D(salty_camera_man, -1, g_kernel3x3)
camera_man_avg3 = cv2.filter2D(salty_camera_man, -1, avg_kernel3x3)
camera_man_median3 = median(salty_camera_man, disk(3))
camera_man_gaussian5 = cv2.filter2D(salty_camera_man, -1, g_kernel5x5)
camera_man_avg5 = cv2.filter2D(salty_camera_man, -1, avg_kernel5x5)
camera_man_median5 = median(salty_camera_man, disk(5))

_, subplt = plt.subplots(2, 3, figsize=(15, 10))
subplt[0][0].imshow(camera_man_gaussian3, cmap='gray')
subplt[0][0].set_title("3x3 Gaussian Filter")
subplt[0][1].imshow(camera_man_avg3, cmap='gray')
subplt[0][1].set_title("3x3 Mean Filter")
subplt[0][2].imshow(camera_man_median3, cmap='gray')
subplt[0][2].set_title("3x3 Median Filter")
subplt[1][0].imshow(camera_man_gaussian5, cmap='gray')
subplt[1][0].set_title("5x5 Gaussian Filter")
subplt[1][1].imshow(camera_man_avg5, cmap='gray')
subplt[1][1].set_title("5x5 Mean Filter")
subplt[1][2].imshow(camera_man_median5, cmap='gray')
subplt[1][2].set_title("5x5 Median Filter")
plt.show()

babon_gaussian3 = cv2.filter2D(salty_babon, -1, g_kernel3x3)
babon_avg3 = cv2.filter2D(salty_babon, -1, avg_kernel3x3)
babon_median3 = median(salty_babon, disk(3))
babon_gaussian5 = cv2.filter2D(salty_babon, -1, g_kernel3x3)
babon_avg5 = cv2.filter2D(salty_babon, -1, avg_kernel5x5)
babon_median5 = median(salty_babon, disk(5))

_, subplt = plt.subplots(2, 3, figsize=(15, 10))
subplt[0][0].imshow(babon_gaussian3, cmap='gray')
subplt[0][0].set_title("3x3 Gaussian Filter")
subplt[0][1].imshow(babon_avg3, cmap='gray')
subplt[0][1].set_title("3x3 Mean Filter")
subplt[0][2].imshow(babon_median3, cmap='gray')
subplt[0][2].set_title("3x3 Median Filter")
subplt[1][0].imshow(babon_gaussian5, cmap='gray')
subplt[1][0].set_title("5x5 Gaussian Filter")
subplt[1][1].imshow(babon_avg5, cmap='gray')
subplt[1][1].set_title("5x5 Mean Filter")
subplt[1][2].imshow(babon_median5, cmap='gray')
subplt[1][2].set_title("5x5 Median Filter")
plt.show()

# Gaussian

camera_man_gaussian3 = cv2.filter2D(gaussian_camera_man, -1, g_kernel3x3)
camera_man_avg3 = cv2.filter2D(gaussian_camera_man, -1, avg_kernel3x3)
camera_man_median3 = median(gaussian_camera_man, disk(3))
camera_man_gaussian5 = cv2.filter2D(gaussian_camera_man, -1, g_kernel5x5)
camera_man_avg5 = cv2.filter2D(gaussian_camera_man, -1, avg_kernel5x5)
camera_man_median5 = median(gaussian_camera_man, disk(5))

_, subplt = plt.subplots(2, 3, figsize=(15, 10))
subplt[0][0].imshow(camera_man_gaussian3, cmap='gray')
subplt[0][0].set_title("3x3 Gaussian Filter")
subplt[0][1].imshow(camera_man_avg3, cmap='gray')
subplt[0][1].set_title("3x3 Mean Filter")
subplt[0][2].imshow(camera_man_median3, cmap='gray')
subplt[0][2].set_title("3x3 Median Filter")
subplt[1][0].imshow(camera_man_gaussian5, cmap='gray')
subplt[1][0].set_title("5x5 Gaussian Filter")
subplt[1][1].imshow(camera_man_avg5, cmap='gray')
subplt[1][1].set_title("5x5 Mean Filter")
subplt[1][2].imshow(camera_man_median5, cmap='gray')
subplt[1][2].set_title("5x5 Median Filter")
plt.show()

babon_gaussian3 = cv2.filter2D(gaussian_babon, -1, g_kernel3x3)
babon_avg3 = cv2.filter2D(gaussian_babon, -1, avg_kernel3x3)
babon_median3 = median(gaussian_babon, disk(3))
babon_gaussian5 = cv2.filter2D(gaussian_babon, -1, g_kernel5x5)
babon_avg5 = cv2.filter2D(gaussian_babon, -1, avg_kernel5x5)
babon_median5 = median(gaussian_babon, disk(5))

_, subplt = plt.subplots(2, 3, figsize=(15, 10))
subplt[0][0].imshow(babon_gaussian3, cmap='gray')
subplt[0][0].set_title("3x3 Gaussian Filter")
subplt[0][1].imshow(babon_avg3, cmap='gray')
subplt[0][1].set_title("3x3 Mean Filter")
subplt[0][2].imshow(babon_median3, cmap='gray')
subplt[0][2].set_title("3x3 Median Filter")
subplt[1][0].imshow(babon_gaussian5, cmap='gray')
subplt[1][0].set_title("5x5 Gaussian Filter")
subplt[1][1].imshow(babon_avg5, cmap='gray')
subplt[1][1].set_title("5x5 Mean Filter")
subplt[1][2].imshow(babon_median5, cmap='gray')
subplt[1][2].set_title("5x5 Median Filter")
plt.show()
