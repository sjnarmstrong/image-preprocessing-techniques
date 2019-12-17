from BuildingBlocks.AffineTransforms import scale_image
from BuildingBlocks.Interpolation import linear_interpolation
import cv2
import numpy as np
from math import ceil, floor
from matplotlib import pyplot as plt
from os import makedirs


show_images = False
save_images = True
save_directory = "out/Question_5/"
Fig0417_a_Location = "../Datasets/DIP3E_Original_Images_CH04/Fig0417(a)(barbara).tif"


def create_checkerboard_pattern(length, x_blocks, y_blocks):
    pattern = np.array([[0, 255],
                        [255, 0]], dtype=np.uint8)
    out_hold = np.repeat(pattern, length, axis=0)
    out_hold = np.repeat(out_hold, length, axis=1)
    out_hold = np.tile(out_hold, (ceil(y_blocks/2), ceil(x_blocks/2)))
    if x_blocks % 2 == 1:
        out_hold = out_hold[:, :-length]
    if y_blocks % 2 == 1:
        out_hold = out_hold[:-length]
    return out_hold


def get_dist_from_centre(u, v, center_u, center_v):
    d_sq_uv = (u - center_u) ** 2 + (v - center_v) ** 2
    return np.sqrt(d_sq_uv)


def create_circle(size, center_u, center_v, radius):
    img_out = np.zeros(size)
    u_points, v_points = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    dists = get_dist_from_centre(u_points, v_points, center_u, center_v)
    points_to_fill = np.where(dists <= radius)
    img_out[points_to_fill] = 1.0
    return img_out


def create_circular_window(img_shape, window_size):
    image_centre_x = (img_shape[1]-1)/2.0
    image_centre_y = (img_shape[0]-1)/2.0
    #out_window = np.zeros(img_shape, dtype=np.uint8)
    #cv2.circle(out_window, (image_centre_y, image_centre_x), int(window_size[0]/2), 1, -1)
    return create_circle(img_shape, image_centre_x, image_centre_y, ceil((window_size[0]-1)/2))


def create_window(img_shape, window_size):
    image_centre_x = img_shape[1]/2
    image_centre_y = img_shape[0]/2
    start_x = int(image_centre_x - window_size[1]/2)
    start_y = int(image_centre_y - window_size[0]/2)
    end_x = ceil(image_centre_x + window_size[1]/2)
    end_y = ceil(image_centre_y + window_size[0]/2)
    out_window = np.zeros(img_shape, dtype=np.uint8)
    out_window[start_y:end_y, start_x:end_x] = 1.0
    return out_window


def filter_image(img, scale_y, scale_x, window_type=create_window):
    img_fft = np.fft.fft2(img)
    img_fft_shifted = np.fft.fftshift(img_fft)
    window = window_type(img.shape, (img.shape[0] * scale_y, img.shape[1] * scale_x))

    img_filtered_fft_shifted = img_fft_shifted * window
    img_filtered_fft = np.fft.ifftshift(img_filtered_fft_shifted)
    ret_img_imag_real = np.fft.ifft2(img_filtered_fft)
    return (np.absolute(ret_img_imag_real),
            20*np.log(np.absolute(img_fft_shifted)))

"""
Create Images as shown in fig 4.16
"""

fig416a = create_checkerboard_pattern(16, 64, 48)
fig416b = create_checkerboard_pattern(6, 96, 54)
fig416c = scale_image(fig416a, (0.9174/16, 0.9174/16))
fig416d = scale_image(fig416b, (0.4798/6, 0.4798/6))

if save_images:
    makedirs(save_directory, exist_ok=True)
    cv2.imwrite(save_directory+"fig416a.png", fig416a[:16*3, :16*4])
    cv2.imwrite(save_directory+"fig416b.png", fig416b[:6*8, :6*16])
    cv2.imwrite(save_directory+"fig416c.png", fig416c[:16*3, :16*4])
    cv2.imwrite(save_directory+"fig416d.png", fig416d[:6*8, :6*16])

if show_images:
    cv2.imshow("fig416a", fig416a[:16*3, :16*4])
    cv2.imshow("fig416b", fig416b[:6*8, :6*16])
    cv2.imshow("fig416c", fig416c[:16*3, :16*4])
    cv2.imshow("fig416d", fig416d[:6*8, :6*16])
    while cv2.waitKey(1) == -1:
        pass
    cv2.destroyAllWindows()


"""
Create Images as shown in fig 4.17
"""

fig417a = cv2.imread(Fig0417_a_Location, cv2.IMREAD_GRAYSCALE)
fig417b = scale_image(fig417a, (0.5, 0.5))

if save_images:
    cv2.imwrite(save_directory+"fig417a.png", fig417a)
    cv2.imwrite(save_directory+"fig417b.png", fig417b)

if show_images:
    cv2.imshow("fig417a", fig417a)
    cv2.imshow("fig417b", fig417b)
    while cv2.waitKey(1) == -1:
        pass
    cv2.destroyAllWindows()

"""
Create filtered images Checkerboard
"""

fig416a_filtered, fig416a_fft = filter_image(fig416a, 0.0573375, 0.0573375)
fig416b_filtered, fig416b_fft = filter_image(fig416b, 0.07996666666666667, 0.07996666666666667)
fig416c_filtered = scale_image(fig416a_filtered, (0.9174/16, 0.9174/16))
fig416d_filtered = scale_image(fig416b_filtered, (0.4798/6, 0.4798/6))

if save_images:
    cv2.imwrite(save_directory+"fig416a_filtered.png", cv2.convertScaleAbs(fig416a_filtered)[:16*3, :16*4])
    cv2.imwrite(save_directory+"fig416b_filtered.png", cv2.convertScaleAbs(fig416b_filtered)[:6*8, :6*16])
    cv2.imwrite(save_directory+"fig416a_fft.png", cv2.convertScaleAbs(fig416a_fft))
    cv2.imwrite(save_directory+"fig416b_fft.png", cv2.convertScaleAbs(fig416b_fft))
    cv2.imwrite(save_directory+"fig416c_filtered.png", cv2.convertScaleAbs(fig416c_filtered))
    cv2.imwrite(save_directory+"fig416d_filtered.png", cv2.convertScaleAbs(fig416d_filtered))

if show_images:
    cv2.imshow("fig416a_filtered", cv2.convertScaleAbs(fig416a_filtered))
    cv2.imshow("fig416b_filtered", cv2.convertScaleAbs(fig416b_filtered))
    cv2.imshow("fig416a_fft", cv2.convertScaleAbs(fig416a_fft))
    cv2.imshow("fig416b_fft", cv2.convertScaleAbs(fig416b_fft))
    cv2.imshow("fig416c_filtered", cv2.convertScaleAbs(fig416c_filtered))
    cv2.imshow("fig416d_filtered", cv2.convertScaleAbs(fig416d_filtered))
    while cv2.waitKey(1) == -1:
        pass
    cv2.destroyAllWindows()

"""
Create filtered images Image
"""

fig417a_filtered, fig417a_fft = filter_image(fig417a, 0.5, 0.5)
fig417b_filtered = scale_image(fig417a_filtered, (0.5, 0.5))

if save_images:
    cv2.imwrite(save_directory+"fig417a_fft.png", cv2.convertScaleAbs(fig417a_fft))
    cv2.imwrite(save_directory+"fig417a_filtered.png", cv2.convertScaleAbs(fig417a_filtered))
    cv2.imwrite(save_directory+"fig417b_filtered.png", cv2.convertScaleAbs(fig417b_filtered))

if show_images:
    cv2.imshow("fig417a_fft.png", cv2.convertScaleAbs(fig417a_fft))
    cv2.imshow("fig417a_filtered", cv2.convertScaleAbs(fig417a_filtered))
    cv2.imshow("fig417b_filtered", cv2.convertScaleAbs(fig417b_filtered))

    while cv2.waitKey(1) == -1:
        pass
    cv2.destroyAllWindows()

"""
Create filtered images Image Circ window
"""

fig417a_filtered_circ, _ = filter_image(fig417a, 0.4, 0.4, create_circular_window)
fig417b_filtered_circ = scale_image(fig417a_filtered_circ, (0.5, 0.5))

if save_images:
    cv2.imwrite(save_directory+"fig417a_filtered_circ.png", cv2.convertScaleAbs(fig417a_filtered_circ))
    cv2.imwrite(save_directory+"fig417b_filtered_circ.png", cv2.convertScaleAbs(fig417b_filtered_circ))

if show_images:
    cv2.imshow("fig417a_filtered_circ", cv2.convertScaleAbs(fig417a_filtered_circ))
    cv2.imshow("fig417b_filtered_circ", cv2.convertScaleAbs(fig417b_filtered_circ))

    while cv2.waitKey(1) == -1:
        pass
    cv2.destroyAllWindows()
