import cv2
import matplotlib.pyplot as plt
import numpy as np

fig_w = 20
fig_h = 6


def aggregate(img_x, img_y):
    img_x = img_x.astype(np.uint32)
    img_y = img_y.astype(np.uint32)
    return np.sqrt((img_x ** 2) + (img_y ** 2)).astype(np.uint8)


kodim = cv2.imread("kodim05.png", cv2.IMREAD_UNCHANGED)
plt.imshow(kodim, cmap='gray')
plt.show()

y_gate = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
x_gate = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
ddepth = cv2.CV_8U
border = cv2.BORDER_REPLICATE
y_kodim = cv2.filter2D(kodim, ddepth, y_gate, borderType=border)
x_kodim = cv2.filter2D(kodim, ddepth, x_gate, borderType=border)

_, subplt = plt.subplots(1, 2, figsize=(fig_w, fig_h))
subplt[0].imshow(y_kodim, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(x_kodim, cmap='gray')
subplt[1].set_title("Filtered Image")
plt.show()

plt.imshow(aggregate(x_kodim, y_kodim), cmap='gray')
plt.show()

F = np.array(
    [[-0.0052625, -0.0173466, -0.0427401, -0.0768961, -0.957739, -0.0696751, 0, 0.6696751, 0.0957739, 0.0768961,
     0.0427401, 0.0173466, 0.0052625]])

Fy_kodim = cv2.filter2D(kodim, ddepth, F.transpose(), borderType=border)
Fx_kodim = cv2.filter2D(kodim, ddepth, F, borderType=border)

_, subplt = plt.subplots(1, 2, figsize=(fig_w, fig_h))
subplt[0].imshow(Fy_kodim, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(Fx_kodim, cmap='gray')
subplt[1].set_title("Filtered Image")
plt.show()

plt.imshow(aggregate(Fx_kodim, Fy_kodim), cmap='gray')
plt.show()
