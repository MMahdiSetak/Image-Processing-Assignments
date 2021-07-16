import cv2
import matplotlib.pyplot as plt
import numpy as np


def watermark(src, target, c):
    src = cv2.dct(src)
    target = cv2.dct(target)
    img = np.copy(src)
    img[0:target.shape[0], 0:target.shape[1]] += target * c
    return cv2.idct(img)


barbara = cv2.imread("barbara.tif", cv2.IMREAD_GRAYSCALE).astype("float64")
plt.imshow(barbara, cmap='gray')
plt.show()

mask = np.ones((300, 200))
mask[50:250, 50:150] = np.zeros((200, 100))

c = [1, 10, 100, 1000]
for i in c:
    watermarked = watermark(barbara, mask, i)
    plt.imshow(watermarked, cmap='gray')
    plt.title(f"c = {i}")
    plt.show()
