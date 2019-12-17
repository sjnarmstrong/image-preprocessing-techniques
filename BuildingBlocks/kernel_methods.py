import numpy as np
import cv2


def uniform_neighborhood_averaging(src, kernel_size=3):
    return cv2.filter2D(src, -1, np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size))


def gaussian_filtering(src, kernel_size=3, sigma_x=0.0, sigma_y=0.0):
    return cv2.GaussianBlur(src, (kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y)


def min_filtering(src, kernel_size=3):
    return cv2.erode(src, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))


def max_filtering(src, kernel_size=3):
    return cv2.dilate(src, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))


def median_filtering(src, kernel_size=3):
    return cv2.medianBlur(src, kernel_size)


"""
image_location = "../../Datasets/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif"

img = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)

img_out_1 = uniform_neighborhood_averaging(img, 5)
img_out_2 = uniform_neighborhood_averaging(img, 9)
img_out_3 = gaussian_filtering(img, 5)
img_out_4 = gaussian_filtering(img, 9, 1, 1)
img_out_5 = gaussian_filtering(img, 13, 0.1, 8)
img_out_6 = min_filtering(img, 3)
img_out_7 = min_filtering(img, 9)
img_out_8 = max_filtering(img, 3)
img_out_9 = max_filtering(img, 9)
img_out_10 = median_filtering(img, 3)
img_out_11 = median_filtering(img, 9)

cv2.imshow("img", img)
cv2.imshow("img Out 1", img_out_1)
cv2.imshow("img Out 2", img_out_2)
cv2.imshow("img Out 3", img_out_3)
cv2.imshow("img Out 4", img_out_4)
cv2.imshow("img Out 5", img_out_5)
cv2.imshow("img Out 6", img_out_6)
cv2.imshow("img Out 7", img_out_7)
cv2.imshow("img Out 8", img_out_8)
cv2.imshow("img Out 9", img_out_9)
cv2.imshow("img Out 10", img_out_10)
cv2.imshow("img Out 11", img_out_11)
while cv2.waitKey(1) == -1:
    pass
cv2.destroyAllWindows()
"""
