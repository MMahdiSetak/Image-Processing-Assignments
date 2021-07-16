import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

imgs = [cv2.imread("StLouisArchMultExpCDR.jpg", cv2.IMREAD_UNCHANGED).astype("float64"),
        cv2.imread("StLouisArchMultExpEV+1.51.jpg", cv2.IMREAD_UNCHANGED).astype("float64"),
        cv2.imread("StLouisArchMultExpEV+4.09.jpg", cv2.IMREAD_UNCHANGED).astype("float64"),
        cv2.imread("StLouisArchMultExpEV-1.82.jpg", cv2.IMREAD_UNCHANGED).astype("float64"),
        cv2.imread("StLouisArchMultExpEV-4.72.jpg", cv2.IMREAD_UNCHANGED).astype("float64")]

img1 = np.mean(imgs, dtype=np.float64, axis=0)
img1 = cv2.cvtColor(img1.astype("uint8"), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(29, 22))
plt.imshow(img1)
plt.show()

cA = np.empty((5, 3), dtype=object)
cH = np.empty((5, 3), dtype=object)
cV = np.empty((5, 3), dtype=object)
cD = np.empty((5, 3), dtype=object)

for i in range(5):
    for j in range(3):
        cA[i][j], (cH[i][j], cV[i][j], cD[i][j]) = pywt.dwt2(imgs[i][:, :, j], 'haar')

CA = np.empty(3, dtype=object)
CH = np.empty(3, dtype=object)
CV = np.empty(3, dtype=object)
CD = np.empty(3, dtype=object)

for i in range(3):
    CA[i] = np.mean(cA[:, i].tolist(), dtype=np.float64, axis=0)
    CH[i] = np.max(cH[:, i].tolist(), axis=0)
    CV[i] = np.max(cV[:, i].tolist(), axis=0)
    CD[i] = np.max(cD[:, i].tolist(), axis=0)

img2 = np.copy(img1)
for i in range(3):
    img2[:, :, i] = pywt.idwt2((CA[i], (CH[i], CV[i], CD[i])), 'haar')

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(29, 22))
plt.imshow(img2)
plt.show()
