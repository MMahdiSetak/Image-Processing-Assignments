import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_comparator(hist, image):
    image_hist = cv2.calcHist([image], [0], None, [256], [0, 255])
    return np.abs(hist - image_hist).sum() / (image.shape[0] * image.shape[1])


def search_in_image(source, target, step):
    height = range(0, source.shape[0] - target.shape[0], step)
    width = range(0, source.shape[1] - target.shape[1], step)
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 255])
    mini = -1
    ans = (-1, -1)
    for i in height:
        for j in width:
            cropped_image = np.array(source)[i:i + target.shape[0], j:j + target.shape[1], :]
            temp_res = hist_comparator(target_hist, cropped_image)
            if temp_res < mini or mini == -1:
                mini = temp_res
                ans = (i, j)
    return np.array(source)[ans[0]:ans[0] + target.shape[0], ans[1]:ans[1] + target.shape[1], :]


source_image = cv2.imread("messi5.jpg", cv2.IMREAD_UNCHANGED)
target_image = cv2.imread("ball.png", cv2.IMREAD_UNCHANGED)

# plt.imshow(source_image)

'''
cv2.imshow("chert", source_image)
cv2.imshow("pert", target_image)
cv2.waitKey(0)
'''

# hist = cv2.calcHist([source_image], [0], None, [256], [0, 255])
# plt.figure()
# plt.plot(hist)
# plt.show()
detected_image = search_in_image(source_image, target_image, 1)
plt.imshow(detected_image)
plt.show()

# plt.waitforbuttonpress()
