import numpy as np
from BuildingBlocks.Interpolation import nearest_interpolation, linear_interpolation


def scale_image(src, scale_vector, x_range=None, y_range=None, interpolation_func=nearest_interpolation):
    if y_range is None:
        y_range = 0, len(src)
    if x_range is None:
        x_range = 0, len(src[0])

    region_x, region_y = np.meshgrid(np.arange(x_range[0], x_range[1], 1.0/scale_vector[0], dtype=np.float32),
                                     np.arange(y_range[0], y_range[1], 1.0/scale_vector[1], dtype=np.float32))

    return interpolation_func(src, region_x, region_y)


def translate_image(src, translation_vector, x_range=None, y_range=None, interpolation_func=nearest_interpolation):
    if y_range is None:
        y_range = 0, len(src)
    if x_range is None:
        x_range = 0, len(src[0])

    region_x, region_y = np.meshgrid(np.arange(x_range[0], x_range[1], dtype=np.float32),
                                     np.arange(y_range[0], y_range[1], dtype=np.float32))

    region_x = region_x - translation_vector[0]
    region_y = region_y - translation_vector[1]

    return interpolation_func(src, region_x, region_y)


def rotate_image(src, theta, origin=(0, 0), x_range=None, y_range=None, interpolation_func=nearest_interpolation):
    if y_range is None:
        y_range = 0, len(src)
    if x_range is None:
        x_range = 0, len(src[0])

    region_x, region_y = np.meshgrid(np.arange(x_range[0], x_range[1], dtype=np.float32),
                                     np.arange(y_range[0], y_range[1], dtype=np.float32))

    theta_rad = np.deg2rad(theta)
    nx = np.cos(theta_rad)*(region_x - origin[0]) + np.sin(theta_rad)*(region_y - origin[1]) + origin[0]
    ny = - np.sin(theta_rad)*(region_x - origin[0]) + np.cos(theta_rad)*(region_y - origin[1]) + origin[1]

    return interpolation_func(src, nx, ny)


def get_shear_borders(src, shear_h: float=0, shear_v: float=0):
    x1 = float(len(src[0]))
    x2 = len(src)*shear_h
    x3 = x1+x2
    y1 = len(src[0])*shear_v
    y2 = float(len(src))
    y3 = y1+y2
    return (min(x1, x2, x3, 0.0), max(x1, x2, x3)), (min(y1, y2, y3, 0.0), max(y1, y2, y3))


def shear_image(src, shear_h: float=0, shear_v: float=0, x_range=None, y_range=None, interpolation_func=nearest_interpolation):
    if y_range is None:
        y_range = 0, len(src)
    if x_range is None:
        x_range = 0, len(src[0])

    region_x, region_y = np.meshgrid(np.arange(x_range[0], x_range[1], dtype=np.float32),
                                     np.arange(y_range[0], y_range[1], dtype=np.float32))

    # nx = region_x + shear_v*region_y
    # ny = shear_h*region_x + region_y

    Dnx = region_x - shear_h * region_y
    Dny = region_y - shear_v * region_x
    D = 1 - shear_h * shear_v

    return interpolation_func(src, Dnx/D, Dny/D)


"""
image_location = "../../Datasets/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif"

img = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)

img_out_1 = scale_image(img, (10, 10), (35, 90), (60, 100))
img_out_2 = scale_image(img, (10, 10), (35, 90), (60, 100), linear_interpolation)
img_out_3 = translate_image(img, (-10.1, 1))
img_out_4 = translate_image(img, (50, -20), interpolation_func=linear_interpolation)
img_out_5 = rotate_image(img, 15, (150, 150), (-50, 350), (-50, 350))
img_out_6 = rotate_image(img, -15, (150, 150), (-50, 350), (-50, 350))

img_out_7 = shear_image(img, 0.1, 0, (-50, 350), (-50, 350))
img_out_8 = shear_image(img, 0, 5, (-50, 350), (-50, 350))


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
