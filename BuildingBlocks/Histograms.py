import cv2


def global_hist_eq(src):
    return cv2.equalizeHist(src)


def block_hist_eq(src, tileSize=8, clipLimit=40.0):
    return cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileSize, tileSize)).apply(src)


"""
image_location = "../../Datasets/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif"

img = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)

img_out_1 = global_hist_eq(img)
img_out_2 = block_hist_eq(img)
img_out_3 = block_hist_eq(img, 8, 1.0)
img_out_4 = block_hist_eq(img, 16, 2.0)

cv2.imshow("img", img)
cv2.imshow("img Out 1", cv2.convertScaleAbs(img_out_1))
cv2.imshow("img Out 2", cv2.convertScaleAbs(img_out_2))
cv2.imshow("img Out 3", cv2.convertScaleAbs(img_out_3))
cv2.imshow("img Out 4", cv2.convertScaleAbs(img_out_4))
while cv2.waitKey(1) == -1:
    pass
cv2.destroyAllWindows()
"""
