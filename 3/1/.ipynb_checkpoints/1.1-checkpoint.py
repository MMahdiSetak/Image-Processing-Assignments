import cv2
import matplotlib.pyplot as plt
import numpy as np

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


s = 4
_, subplt = plt.subplots(4, 2, figsize=(15, 30))
for i in range(4):
    s *= 2
    freq_mandrill = block_proc(mandrill, (s, s), cv2.dct)
    subplt[i][0].imshow(freq_mandrill, cmap='jet')
    subplt[i][0].title.set_text(f"Block {s}x{s}")
    filtered_mandrill = block_proc(freq_mandrill, (s, s), lambda img: img * get_mask(s))
    compressed_mandrill = block_proc(filtered_mandrill, (s, s), cv2.idct)
    subplt[i][1].imshow(compressed_mandrill, cmap='gray')
    subplt[i][1].title.set_text(f"PSNR: {cv2.PSNR(mandrill, compressed_mandrill)}")
plt.show()
