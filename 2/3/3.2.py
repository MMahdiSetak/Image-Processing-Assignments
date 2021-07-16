import cv2
import numpy as np
from matplotlib import pyplot as plt

lena = cv2.imread('lena512.bmp', cv2.IMREAD_GRAYSCALE).astype('float64')
plt.imshow(lena, cmap='gray')
plt.title('Original')
plt.show()

lena_freq = np.fft.fft2(lena)

center = (int(lena.shape[0] / 2), int(lena.shape[1] / 2))
Y, X = np.ogrid[:lena.shape[0], :lena.shape[1]]
dist_from_center = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)

stp = 20
_, subplt = plt.subplots(int(lena.shape[0] / 2 / stp) + 1, 2, figsize=(20, 100))
for i in range(5, int(lena.shape[0] / 2), stp):
    index = int(i / stp)
    midIndex = np.ceil(int(lena.shape[0]) / stp / 4)
    radius = i
    if index <= midIndex:
        radius -= index * int(stp / 2)
    mask = dist_from_center <= radius
    subplt[index][0].imshow(mask, cmap='gray')
    subplt[index][0].title.set_text(f"Radius: {radius}")
    mask = np.fft.fftshift(mask)
    result = np.fft.ifft2(lena_freq * mask)
    result = np.absolute(result)
    subplt[index][1].imshow(result, cmap='gray')
    subplt[index][1].title.set_text(f"PSNR: {cv2.PSNR(lena, result)}")
plt.show()

stp = 40
_, subplt = plt.subplots(int(lena.shape[0] / stp) + 1, 2, figsize=(20, 100))
for i in range(5, lena.shape[0], stp):
    index = int(i / stp)
    midIndex = np.ceil(int(lena.shape[0]) / stp / 2)
    radius = i
    if index <= midIndex:
        radius -= index * int(stp / 2)
    x = cv2.getGaussianKernel(lena.shape[0], radius)
    mask = x * x.T
    subplt[index][0].imshow(mask, cmap='gray')
    subplt[index][0].title.set_text(f"Radius: {radius}")
    mask = np.fft.fftshift(mask)
    result = np.fft.ifft2(lena_freq * mask)
    result = np.absolute(result)
    subplt[index][1].imshow(result, cmap='gray')
    subplt[index][1].title.set_text(f"PSNR: {cv2.PSNR(lena, result)}")
plt.show()
