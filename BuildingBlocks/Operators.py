import cv2


def Sobel(src, dx=1, dy=1, k_size=3, d_depth=cv2.CV_64F):
    return cv2.Sobel(src, d_depth, dx, dy, ksize=k_size)


def Laplacian(src, k_size=3, d_depth=cv2.CV_64F):
    return cv2.Laplacian(src, ddepth=d_depth, ksize=k_size)

"""
image_location = "../../Datasets/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif"

img = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)

img_out_1 = Sobel(img, 0, 1)
img_out_2 = Sobel(img, 1, 0)
img_out_3 = Laplacian(img, 3)
img_out_4 = Laplacian(img, 5)

cv2.imshow("img", img)
cv2.imshow("img Out 1", cv2.convertScaleAbs(img_out_1))
cv2.imshow("img Out 2", cv2.convertScaleAbs(img_out_2))
cv2.imshow("img Out 3", cv2.convertScaleAbs(img_out_3))
cv2.imshow("img Out 4", cv2.convertScaleAbs(img_out_4))
while cv2.waitKey(1) == -1:
    pass
cv2.destroyAllWindows()
"""
