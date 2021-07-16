import cv2
import numpy as np
from matplotlib import pyplot as plt

checker_board = cv2.imread('CheckerBoard.png', cv2.IMREAD_GRAYSCALE)
vertical = cv2.imread('Vertical.png', cv2.IMREAD_GRAYSCALE)

vertical_freq = np.fft.fft2(vertical)


