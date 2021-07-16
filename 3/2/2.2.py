import cv2
import matplotlib.pyplot as plt
import numpy as np


def watermark(src, target, c):
    src = cv2.dct(src)
    target = cv2.dct(target)
    img = np.copy(src)
    img[0:target.shape[0], 0:target.shape[1]] += target * c
    return cv2.idct(img)


barbara = cv2.imread("scaled_barbara.tif", cv2.IMREAD_GRAYSCALE).astype("float64")
plt.imshow(barbara, cmap='gray')
plt.title("Original Image")
plt.show()

mask = np.ones((300, 200))
mask[50:250, 50:150] = np.zeros((200, 100))

result = np.copy(barbara)
bs = 32
for i in range(0, barbara.shape[0], bs):
    for j in range(0, barbara.shape[1], bs):
        result[i:i + bs, j:j + bs] = watermark(barbara[i:i + bs, j:j + bs],
                                               mask[i // 2:(i + bs) // 2, j // 2:(j + bs) // 2], 300)

plt.imshow(result, cmap='gray')
plt.title("Result")
plt.show()
