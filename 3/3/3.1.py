import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise

messi = cv2.imread("messi5.jpg", cv2.IMREAD_GRAYSCALE)
barbara = cv2.imread("barbara.tif", cv2.IMREAD_GRAYSCALE)

_, subplt = plt.subplots(1, 2, figsize=(15, 12))
subplt[0].imshow(messi, cmap='gray')
subplt[0].title.set_text("Original Messi")
subplt[1].imshow(barbara, cmap='gray')
subplt[1].title.set_text("Original Barbara")
plt.show()

gaussian_messi = random_noise(messi, mode='gaussian', var=0.1, mean=0)
sp_messi = random_noise(messi, mode='s&p', amount=0.1)
gaussian_barbara = random_noise(barbara, mode='gaussian', var=0.1, mean=0)
sp_barbara = random_noise(barbara, mode='s&p', amount=0.1)


def show_result():
    _, subplt = plt.subplots(2, 2, figsize=(15, 12))
    subplt[0][0].imshow(gaussian_messi, cmap='gray')
    subplt[0][0].title.set_text(f"Gaussian Noise(PSNR: {cv2.PSNR(messi.astype('float64'), gaussian_messi)})")
    subplt[0][1].imshow(sp_messi, cmap='gray')
    subplt[0][1].title.set_text(f"Salt & Pepper Noise(PSNR: {cv2.PSNR(messi.astype('float64'), sp_messi)})")
    subplt[1][0].imshow(gaussian_barbara, cmap='gray')
    subplt[1][0].title.set_text(f"Gaussian Noise(PSNR: {cv2.PSNR(barbara.astype('float64'), gaussian_barbara)})")
    subplt[1][1].imshow(sp_barbara, cmap='gray')
    subplt[1][1].title.set_text(f"Salt & Pepper Noise(PSNR: {cv2.PSNR(barbara.astype('float64'), sp_barbara)})")
    plt.show()


show_result()


def dct_noise_filter(src, s):
    img = np.copy(src)
    h = img.shape[0]
    w = img.shape[1]
    mask = np.zeros((h, w))
    mask[:h // s, :w // s] = np.ones((h // s, w // s))
    img = cv2.dct(img)
    img *= mask
    img = cv2.idct(img)
    return img


filter_factor = 4
gaussian_messi = dct_noise_filter(gaussian_messi, filter_factor)
sp_messi = dct_noise_filter(sp_messi, filter_factor)
gaussian_barbara = dct_noise_filter(gaussian_barbara, filter_factor)
sp_barbara = dct_noise_filter(sp_barbara, filter_factor)

show_result()
