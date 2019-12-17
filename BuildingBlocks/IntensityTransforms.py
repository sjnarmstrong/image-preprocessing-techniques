import numpy as np
#import cv2


def log_transform(src, c, prescale=1.0, base=np.e):
    out = c*np.log(1.0+src/prescale)/np.log(base)
    #return cv2.convertScaleAbs(out)
    return out


def gamma_transform(src, c, gamma):
    out = c*src**gamma
    #return cv2.convertScaleAbs(out)
    return out


def contrast_stretching(src, p0, p1, L=256):
    out = np.interp(src, [0, p0[0], p1[0], L-1], [0, p0[1], p1[1], L-1])
    #return cv2.convertScaleAbs(out)
    return out


def bit_level_slicing(src, mask):
    hold_img = src.copy()
    hold_img.shape += (1,)
    hold_img = np.unpackbits(hold_img, axis=2) * mask
    hold_img = np.packbits(hold_img, axis=2)
    hold_img.shape = hold_img.shape[:-1]
    return hold_img


def intensity_level_slicing_two_tones(src, grey_range, off_value=0, on_value=255):
    out_img = np.ones(src.shape, src.dtype)*off_value
    on_indc = np.where((src.flat > grey_range[0]) & (src.flat <= grey_range[1]))
    out_img.flat[on_indc] = on_value
    return out_img


def intensity_level_slicing_scaled(src, grey_range, scale=1.0, on_value=255):
    out_img = src*scale
    on_indc = np.where((src.flat > grey_range[0]) & (src.flat <= grey_range[1]))
    out_img.flat[on_indc] = on_value
    #return cv2.convertScaleAbs(out_img)
    return out_img

"""
image_location = "../../Datasets/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif"

img = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)

img_out_1 = log_transform(img, 255, 25.5, 10)
img_out_2 = log_transform(img, 25, 10)
img_out_3 = gamma_transform(img, 9.8, 0.6)
img_out_3 = gamma_transform(img, 9.8, 0.7)
img_out_4 = gamma_transform(img, 5.6, 0.4)
img_out_5 = gamma_transform(img, 5.6, 0.3)
img_out_6 = gamma_transform(img, 0.11, 2)
img_out_7 = gamma_transform(img, 0.11, 10)
img_out_8 = gamma_transform(img, 0.11, 25)

img_out_3 = contrast_stretching(img, (np.min(img), 0), (np.max(img), 255))
img_out_4 = contrast_stretching(img, (np.min(img), 20), (np.max(img), 255))
img_out_5 = contrast_stretching(img, (np.min(img), 40), (np.max(img), 255))
img_out_6 = contrast_stretching(img, (np.min(img), 50), (np.max(img), 10))
img_out_7 = contrast_stretching(img, (np.min(img), 40), (np.max(img), 25))
img_out_8 = contrast_stretching(img, (np.min(img), 30), (np.max(img), 10))

img_out_3 = bit_level_slicing(img, [0, 0, 0, 0, 0, 1, 1, 1])
img_out_4 = bit_level_slicing(img, [0, 0, 0, 0, 1, 0, 0, 0])
img_out_5 = bit_level_slicing(img, [0, 0, 0, 1, 0, 0, 0, 0])
img_out_6 = bit_level_slicing(img, [0, 0, 1, 0, 0, 0, 0, 0])
img_out_7 = bit_level_slicing(img, [0, 1, 0, 0, 0, 0, 0, 0])
img_out_8 = bit_level_slicing(img, [1, 0, 0, 0, 0, 0, 0, 0])


img_out_1 = intensity_level_slicing_scaled(img, (0, 20), scale=0.2, on_value=125)
img_out_2 = intensity_level_slicing_scaled(img, (20, 40), scale=0.2, on_value=125)
img_out_3 = intensity_level_slicing_scaled(img, (40, 80), scale=0.2, on_value=125)
img_out_4 = intensity_level_slicing_scaled(img, (80, 100), scale=0.2, on_value=125)
img_out_5 = intensity_level_slicing_scaled(img, (100, 120), scale=0.2, on_value=125)
img_out_6 = intensity_level_slicing_scaled(img, (120, 180), scale=0.2, on_value=125)
img_out_7 = intensity_level_slicing_scaled(img, (180, 210), scale=0.2, on_value=125)
img_out_8 = intensity_level_slicing_scaled(img, (210, 255), scale=0.2, on_value=125)

cv2.imshow("img", img)
cv2.imshow("img Out 1", img_out_1)
cv2.imshow("img Out 2", img_out_2)
cv2.imshow("img Out 3", img_out_3)
cv2.imshow("img Out 4", img_out_4)
cv2.imshow("img Out 5", img_out_5)
cv2.imshow("img Out 6", img_out_6)
cv2.imshow("img Out 7", img_out_7)
cv2.imshow("img Out 8", img_out_8)
while cv2.waitKey(1) == -1:
    pass
cv2.destroyAllWindows()

"""
