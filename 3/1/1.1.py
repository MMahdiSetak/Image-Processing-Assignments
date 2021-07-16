import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

mandrill = cv2.imread("mandril.tiff", cv2.IMREAD_GRAYSCALE).astype("float64")
plt.imshow(mandrill, cmap='gray')
plt.title("Original Image")
plt.show()


def block_proc(image, size, function):
    ret = np.copy(image)
    for i in range(0, ret.shape[0], size[0]):
        for j in range(0, ret.shape[1], size[1]):
            ret[i:i + size[0], j:j + size[1]] = function(ret[i:i + size[0], j:j + size[1]])
    return ret


def get_mask(size):
    ret = np.zeros((size, size))
    ret[0:size // 2, 0:size // 2] = np.ones((size // 2, size // 2))
    return ret


times = []
PSNRs = []
s = 4
_, subplt = plt.subplots(4, 2, figsize=(15, 30))
for i in range(4):
    s *= 2
    tic = time.perf_counter()
    freq_mandrill = block_proc(mandrill, (s, s), cv2.dct)
    filtered_mandrill = block_proc(freq_mandrill, (s, s), lambda img: img * get_mask(s))
    compressed_mandrill = block_proc(filtered_mandrill, (s, s), cv2.idct)
    toc = time.perf_counter()
    times.append(f"{toc - tic:.6f}")
    subplt[i][0].imshow(freq_mandrill, cmap='jet')
    subplt[i][0].title.set_text(f"Block {s}x{s}")
    subplt[i][1].imshow(compressed_mandrill, cmap='gray')
    PSNRs.append(f"{cv2.PSNR(mandrill, compressed_mandrill):.6f}")
    subplt[i][1].title.set_text(f"PSNR: {PSNRs[-1]}")
plt.show()

data = [times, PSNRs]
row = ["Execution Time", "PSNR"]
col = ["8x8 Blocks", "16x16 Blocks", "32x32 Blocks", "64x64 Blocks"]

plt.figure(figsize=(8, 3))
table = plt.table(data, rowLabels=row, colLabels=col, loc='center right')
table.scale(0.9, 2)
plt.axis('off')
plt.show()
