import cv2
import numpy as np
from matplotlib import pyplot as plt

checker_board = cv2.imread('CheckerBoard.png', cv2.IMREAD_GRAYSCALE)
checker_board_freq = np.fft.fft2(checker_board)
vertical = cv2.imread('Vertical.png', cv2.IMREAD_GRAYSCALE)
vertical_freq = np.fft.fft2(vertical)

v_min = -1000
v_max = 1000

plt.imshow(vertical, cmap='gray')
plt.title('Original Vertical')
plt.show()
plt.imshow(vertical_freq.real, vmin=v_min, vmax=v_max, cmap='gray')
plt.title('Real')
plt.show()
plt.imshow(vertical_freq.imag, vmin=v_min, vmax=v_max, cmap='gray')
plt.title('Imaginary')
plt.show()
plt.imshow(np.absolute(vertical_freq), vmin=v_min, vmax=v_max, cmap='gray')
plt.title('Absolute')
plt.show()
plt.imshow(np.angle(vertical_freq), vmin=-np.pi, vmax=np.pi, cmap='gray')
plt.title('Angle')
plt.show()


plt.imshow(checker_board, cmap='gray')
plt.title('Original Vertical')
plt.show()
plt.imshow(checker_board_freq.real, vmin=v_min, vmax=v_max, cmap='gray')
plt.title('Real')
plt.show()
plt.imshow(checker_board_freq.imag, vmin=v_min, vmax=v_max, cmap='gray')
plt.title('Imaginary')
plt.show()
plt.imshow(np.absolute(checker_board_freq), vmin=v_min, vmax=v_max, cmap='gray')
plt.title('Absolute')
plt.show()
plt.imshow(np.angle(checker_board_freq), vmin=-np.pi, vmax=np.pi, cmap='gray')
plt.title('Angle')
plt.show()