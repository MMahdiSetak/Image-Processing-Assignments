import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

mandrill = cv2.imread("mandril.tiff", cv2.IMREAD_GRAYSCALE).astype("float64")
plt.imshow(mandrill, cmap='gray')
plt.show()


def get_mask(size):
    ret = np.zeros((size, size))
    ret[0:size // 2, 0:size // 2] = np.ones((size // 2, size // 2))
    return ret


class Block:
    i: int
    j: int
    sharpness: float
    size: int


masks = {8: get_mask(8), 16: get_mask(16), 64: get_mask(64)}


def show_transformed_image():
    img = np.copy(mandrill)
    for block in blocks:
        i = block.i
        ii = i + block.size
        j = block.j
        jj = j + block.size
        img[i:ii, j:jj] = cv2.dct(img[i:ii, j:jj])
        img[i:ii, j:jj] = img[i:ii, j:jj] * masks[block.size]
    _, subplt = plt.subplots(1, 2, figsize=(15, 10))
    subplt[0].imshow(img, cmap='jet')
    subplt[0].title.set_text("DCT mode")
    for block in blocks:
        i = block.i
        ii = i + block.size
        j = block.j
        jj = j + block.size
        img[i:ii, j:jj] = cv2.idct(img[i:ii, j:jj])
    PSNRs.append(f"{cv2.PSNR(mandrill, img):.6f}")
    subplt[1].imshow(img, cmap='gray')
    subplt[1].title.set_text(f"PSNR: {PSNRs[-1]}")
    plt.show()


def break_blocks(s, init):
    if not init:
        blocks.sort(key=lambda x: x.sharpness, reverse=True)
    n = 1 if init else len(blocks) // 10
    for i in range(n):
        for ii in range(0, mandrill.shape[0] if init else blocks[i].size, s):
            for jj in range(0, mandrill.shape[1] if init else blocks[i].size, s):
                block = Block()
                block.i = ii if init else ii + blocks[i].i
                block.j = jj if init else jj + blocks[i].j
                block.size = s
                block.sharpness = cv2.Laplacian(mandrill[block.i:block.i + s, block.j:block.j + s], cv2.CV_64F).var()
                blocks.append(block)
    if not init:
        del blocks[0:n]


blocks = []
times = []
PSNRs = []
block_sizes = [64, 16, 8]
total_tic = time.perf_counter()
for s in block_sizes:
    tic = time.perf_counter()
    break_blocks(s, s == 64)
    show_transformed_image()
    toc = time.perf_counter()
    times.append(f"{toc - tic:.6f}")
total_toc = time.perf_counter()
times.append(f"{total_toc - total_tic:.6f}")
PSNRs.append(PSNRs[-1])

data = [times, PSNRs]
row = ["Execution Time", "PSNR"]
col = ["64x64 Blocks", "16x16 Blocks", "8x8 Blocks", "Final Result"]

plt.figure(figsize=(8, 3))
table = plt.table(data, rowLabels=row, colLabels=col, loc='center right')
table.scale(0.9, 2)
plt.axis('off')
plt.show()
