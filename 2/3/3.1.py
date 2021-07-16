import cv2
import numpy as np
from matplotlib import pyplot as plt

lena = cv2.imread('lena512.bmp', cv2.IMREAD_GRAYSCALE)
lena = lena.astype('float64')
plt.imshow(lena, cmap='gray')
plt.title('Original')
plt.show()

horizontal_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 9
vertical_sobel = horizontal_sobel.T

horizontal_filtered_lena = cv2.filter2D(lena, -1, -horizontal_sobel)
vertical_filtered_lena = cv2.filter2D(lena, -1, -vertical_sobel)

_, subplt = plt.subplots(1, 2, figsize=(10, 5))
subplt[0].imshow(horizontal_filtered_lena, cmap='gray')
subplt[0].title.set_text("Horizontal")
subplt[1].imshow(vertical_filtered_lena, cmap='gray')
subplt[1].title.set_text('Vertical')
plt.show()

padded_horizontal_sobel = np.zeros(lena.shape)
padded_horizontal_sobel[0:3, 0:3] = horizontal_sobel
padded_vertical_sobel = np.zeros(lena.shape)
padded_vertical_sobel[0:3, 0:3] = vertical_sobel

lena_freq = np.fft.fft2(lena)
horizontal_sobel_freq = np.fft.fft2(padded_horizontal_sobel)
vertical_sobel_freq = np.fft.fft2(padded_vertical_sobel)

_, subplt = plt.subplots(1, 2, figsize=(10, 5))
subplt[0].imshow(np.absolute(horizontal_sobel_freq), cmap='gray')
subplt[0].title.set_text("Horizontal")
subplt[1].imshow(np.absolute(vertical_sobel_freq), cmap='gray')
subplt[1].title.set_text('Vertical')
plt.show()

horizontal_filtered_lena_freq = lena_freq * horizontal_sobel_freq
vertical_filtered_lena_freq = lena_freq * vertical_sobel_freq

horizontal_filtered_lena_freq = np.fft.ifft2(horizontal_filtered_lena_freq)
vertical_filtered_lena_freq = np.fft.ifft2(vertical_filtered_lena_freq)

_, subplt = plt.subplots(1, 2, figsize=(10, 5))
subplt[0].imshow(np.absolute(horizontal_filtered_lena_freq), cmap='gray')
subplt[0].title.set_text("Horizontal")
subplt[1].imshow(np.absolute(vertical_filtered_lena_freq.real), cmap='gray')
subplt[1].title.set_text('Vertical')
plt.show()

_, subplt = plt.subplots(2, 2, figsize=(10, 10))
subplt[0][0].imshow(horizontal_filtered_lena, cmap='gray')
subplt[0][0].title.set_text("Horizontal in Spatial Mode")
subplt[0][1].imshow(vertical_filtered_lena, cmap='gray')
subplt[0][1].title.set_text('Vertical in Spatial Mode')
subplt[1][0].imshow(horizontal_filtered_lena_freq.real, cmap='gray')
subplt[1][0].title.set_text('Horizontal in Frequency Mode')
subplt[1][1].imshow(vertical_filtered_lena_freq.real, cmap='gray')
subplt[1][1].title.set_text('Vertical in Frequency Mode')
plt.show()
