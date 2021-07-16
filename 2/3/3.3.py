from random import randint
import numpy as np
from matplotlib import pyplot as plt

img = np.zeros((100, 100))
mask = np.ones((10, 10))

base_index = 50 - int(mask.shape[0] / 2)
img[base_index:base_index + mask.shape[0], base_index:base_index + mask.shape[0]] = mask

img_freq = np.fft.fft2(img)
img_freq = np.fft.fftshift(img_freq)

_, subplt = plt.subplots(1, 2, figsize=(10, 5))
subplt[0].imshow(img, cmap='gray')
subplt[0].title.set_text("Centered 10x10 Mask")
subplt[1].imshow(np.absolute(img_freq), cmap='gray')
subplt[1].title.set_text('Transformed')
plt.show()

for _ in range(5):
    x = randint(0, 90)
    x -= 45
    y = randint(0, 90)
    y -= 45
    img = np.zeros((100, 100))
    mask = np.ones((10, 10))
    base_index = 50 - int(mask.shape[0] / 2)
    img[base_index + y:base_index + mask.shape[0] + y, base_index + x:base_index + mask.shape[0] + x] = mask
    img_freq = np.fft.fft2(img)
    img_freq = np.fft.fftshift(img_freq)
    _, subplt = plt.subplots(1, 2, figsize=(10, 5))
    subplt[0].imshow(img, cmap='gray')
    subplt[0].title.set_text("Moved 10x10 Mask")
    subplt[1].imshow(np.absolute(img_freq), cmap='gray')
    subplt[1].title.set_text('Transformed')
    plt.show()

img = np.zeros((100, 100))
for i in range(10, 101, 18):
    mask = np.ones((i, i))
    base_index = 50 - int(mask.shape[0] / 2)
    img[base_index:base_index + mask.shape[0], base_index:base_index + mask.shape[0]] = mask
    img_freq = np.fft.fft2(img)
    img_freq = np.fft.fftshift(img_freq)
    _, subplt = plt.subplots(1, 2, figsize=(10, 5))
    subplt[0].imshow(img, vmin=0, vmax=1, cmap='gray')
    subplt[0].title.set_text(f"Centered {i}x{i} Mask")
    subplt[1].imshow(np.absolute(img_freq), cmap='gray')
    subplt[1].title.set_text('Transformed')
    plt.show()
