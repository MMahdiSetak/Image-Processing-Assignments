import cv2
import matplotlib.pyplot as plt
import numpy as np


def watermark(src, target, c):
    src = cv2.dct(src)
    target = cv2.dct(target)
    img = np.copy(src)
    img[0:target.shape[0], 0:target.shape[1]] += target * c
    # img[0:target.shape[0], 0:target.shape[1]] = (c * img[0:target.shape[0], 0:target.shape[1]] + target) / (c + 1)
    return cv2.idct(img)


barbara = cv2.imread("barbara.tif", cv2.IMREAD_GRAYSCALE).astype("float64")
plt.imshow(barbara, cmap='gray')
plt.show()

mask = np.ones((200, 300)) * 255
mask[50:150, 100:200] = np.zeros((100, 100))

watermarked = watermark(barbara, mask, 1)
plt.imshow(watermarked, cmap='gray')
plt.title("c = 1")
plt.show()
watermarked = watermark(barbara, mask, 10)
plt.imshow(watermarked, cmap='gray')
plt.title("c = 10")
plt.show()
watermarked = watermark(barbara, mask, 100)
plt.imshow(watermarked, cmap='gray')
plt.title("c = 100")
plt.show()
watermarked = watermark(barbara, mask, 1000)
plt.imshow(watermarked, cmap='gray')
plt.title("c = 1000")
plt.show()
