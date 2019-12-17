import cv2


def nearest_interpolation(src, x_coords, y_coords):
    return cv2.remap(src, x_coords, y_coords, cv2.INTER_NEAREST)


def linear_interpolation(src, x_coords, y_coords):
    return cv2.remap(src, x_coords, y_coords, cv2.INTER_LINEAR)


"""
import numpy as np

image_location = "../../Datasets/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif"

img = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
cv2.imshow("img", img)
cv2.waitKey(100)

test_region_x = np.reshape(np.arange(35, 90, 0.05, dtype=np.float32), (1, -1))
test_region_y = np.reshape(np.arange(60, 100, 0.05, dtype=np.float32), (-1, 1))

test_region_x = np.repeat(test_region_x, len(test_region_y), axis=0)
test_region_y = np.repeat(test_region_y, len(test_region_x[0]), axis=1)

img_out_1 = nearest_interpolation(img, test_region_x, test_region_y)
img_out_2 = linear_interpolation(img, test_region_x, test_region_y)

cv2.imshow("img Out 1", img_out_1)
cv2.imshow("img Out 2", img_out_2)
cv2.waitKey(10000)"""

