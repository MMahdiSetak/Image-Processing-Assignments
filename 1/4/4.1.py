import cv2
import matplotlib.pyplot as plt
import numpy as np

fig_w = 30
fig_h = 20

paper = cv2.imread("paper.png", cv2.IMREAD_GRAYSCALE)

gate = np.array([[0, -1, 0],
                 [-1, 5, -1],
                 [0, -1, 0]])
ddepth = cv2.CV_8U
border = cv2.BORDER_REPLICATE
sharped_paper = cv2.filter2D(paper, ddepth, gate, borderType=border)

_, subplt = plt.subplots(1, 2, figsize=(fig_w, fig_h))
subplt[0].imshow(paper, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(sharped_paper, cmap='gray')
subplt[1].set_title("Filtered Image")
plt.show()

size = 5
blur_kernel = np.ones((size, size), dtype=np.uint8) / (size ** 2)
blur_paper = cv2.filter2D(paper, -1, blur_kernel)

intersect_paper = paper - blur_paper

blur_sharped_paper = paper + intersect_paper

_, subplt = plt.subplots(3, 1, figsize=(10, 45))
subplt[0].imshow(blur_paper, cmap='gray')
subplt[0].set_title("Blur Image")
subplt[1].imshow(intersect_paper, cmap='gray')
subplt[1].set_title("Intersected Image")
subplt[2].imshow(blur_sharped_paper, cmap='gray')
subplt[2].set_title("Sharped Image")
plt.show()

_, subplt = plt.subplots(1, 2, figsize=(fig_w, fig_h))
subplt[0].imshow(sharped_paper, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(blur_sharped_paper, cmap='gray')
subplt[1].set_title("Blur Sharped Image")
plt.show()

_, subplt = plt.subplots(1, 2, figsize=(fig_w, fig_h))
subplt[0].imshow(sharped_paper, cmap='gray')
subplt[0].set_title("Laplacian Sharped Image")
subplt[1].imshow(blur_sharped_paper, cmap='gray')
subplt[1].set_title("Blur Sharped Image")
plt.show()
