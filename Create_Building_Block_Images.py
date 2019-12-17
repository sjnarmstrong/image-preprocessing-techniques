from BuildingBlocks.Interpolation import linear_interpolation, nearest_interpolation
from BuildingBlocks.AffineTransforms import scale_image, rotate_image, shear_image, translate_image, get_shear_borders
from BuildingBlocks.Histograms import block_hist_eq, global_hist_eq
from BuildingBlocks.IntensityTransforms import bit_level_slicing, contrast_stretching, gamma_transform, intensity_level_slicing_scaled, intensity_level_slicing_two_tones, log_transform
from BuildingBlocks.kernel_methods import gaussian_filtering, max_filtering, median_filtering, min_filtering, uniform_neighborhood_averaging
from BuildingBlocks.Operators import Sobel, Laplacian
from plotTools import Plotter
import cv2
import numpy as np


show_plots = False
save_plots = True

output_dir_AffineTransforms = "out/Building_blocks/AffineTransforms/"
output_dir_Histograms = "out/Building_blocks/Histograms/"
output_dir_IntensityTransforms = "out/Building_blocks/IntensityTransforms/"
output_dir_kernel_methods = "out/Building_blocks/kernel_methods/"
output_dir_Operators = "out/Building_blocks/Operators/"


Fig0235_c_Location = "../Datasets/DIP3E_Original_Images_CH02/Fig0235(c)(kidney_original).tif"
Fig0236_a_Location = "../Datasets/DIP3E_Original_Images_CH02/Fig0236(a)(letter_T).tif"
Fig0237_a_Location = "../Datasets/DIP3E_Original_Images_CH02/Fig0237(a)(characters test pattern)_POST.tif"

Fig0305_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0305(a)(DFT_no_log).tif"
Fig0308_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0308(a)(fractured_spine).tif"
Fig0309_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0309(a)(washed_out_aerial_image).tif"
Fig0310_b_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0310(b)(washed_out_pollen_image).tif"
Fig0312_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0312(a)(kidney).tif"
Fig0314_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0314(a)(100-dollars).tif"
Fig0320_b_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0320(2)(2nd_from_top).tif"
Fig0326_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0326(a)(embedded_square_noisy_512).tif"
Fig0335_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif"
Fig0340_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0340(a)(dipxe_text).tif"
Fig0343_a_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0343(a)(skeleton_orig).tif"


def safe_read_image(location):
    hold_out = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
    if hold_out is None:
        print("Could not find image at: " + location)
        print("Please ensure that you have extracted the datasets to the relevant locations")
        exit(0)
    return hold_out


fig0235_c_kidney = safe_read_image(Fig0235_c_Location)
fig0236_a_letter_T = safe_read_image(Fig0236_a_Location)
fig0237_a_characters_test_pattern = safe_read_image(Fig0237_a_Location)

fig0305_a_dft = safe_read_image(Fig0305_a_Location)
fig0308_a_fractured_spine = safe_read_image(Fig0308_a_Location)
fig0309_a_arial = safe_read_image(Fig0309_a_Location)
fig0310_b_washedout_pollen = safe_read_image(Fig0310_b_Location)
fig0312_a_kidney = safe_read_image(Fig0312_a_Location)
fig0314_a_dollar_bill = safe_read_image(Fig0314_a_Location)
fig0320_b_washedout_pollen_2 = safe_read_image(Fig0320_b_Location)
fig0326_a_hidden_icons = safe_read_image(Fig0326_a_Location)
fig0335_a_pcb_noise = safe_read_image(Fig0335_a_Location)
fig0340_a_dipxe = safe_read_image(Fig0340_a_Location)
fig0343_a_skeleton = safe_read_image(Fig0343_a_Location)
"""
Create Scale and Interpolation Images
"""
fig0236_a_scale_near = scale_image(fig0236_a_letter_T, (4, 4), interpolation_func=nearest_interpolation)
fig0236_a_scale_bi = scale_image(fig0236_a_letter_T, (4, 4), interpolation_func=linear_interpolation)

Plotter.no_scale_save_plots(save_plots, output_dir_AffineTransforms+"fig0236_a_letter_T.png", fig0236_a_letter_T)
Plotter.no_scale_save_plots(save_plots, output_dir_AffineTransforms+"fig0236_a_scale_near.png", fig0236_a_scale_near)
Plotter.no_scale_save_plots(save_plots, output_dir_AffineTransforms+"fig0236_a_scale_bi.png", fig0236_a_scale_bi)


#_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_scale_near")
#ax.imshow(fig0236_a_scale_near, cmap="gray")
#_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_scale_bi")
#ax.imshow(fig0236_a_scale_bi, cmap="gray")
"""
Create Rotation and Interpolation Images
"""

fig0236_a_rotation_out_near = rotate_image(fig0236_a_letter_T, 21, (int((fig0236_a_letter_T.shape[0] - 1) / 2),
                                                                    int((fig0236_a_letter_T.shape[1] - 1) / 2)),
                                           interpolation_func=nearest_interpolation)
fig0236_a_rotation_out_bi = rotate_image(fig0236_a_letter_T, 21, (int((fig0236_a_letter_T.shape[0] - 1) / 2),
                                                                  int((fig0236_a_letter_T.shape[1] - 1) / 2)),
                                         interpolation_func=linear_interpolation)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_rotation")
ax.imshow(fig0236_a_letter_T, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_rotation_out_near")
ax.imshow(fig0236_a_rotation_out_near, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_rotation_out_bi")
ax.imshow(fig0236_a_rotation_out_bi, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_rotation_out_near_subsec")
ax.imshow(fig0236_a_rotation_out_near[150:200, 220:270], cmap="gray", interpolation="nearest")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_rotation_out_bi_subsec")
ax.imshow(fig0236_a_rotation_out_bi[150:200, 220:270], cmap="gray", interpolation="nearest")

"""
Create ShearImages
"""
x_r, y_r = get_shear_borders(fig0236_a_letter_T, 0.3, 0)
fig0236_a_shear_v = shear_image(fig0236_a_letter_T, 0.3, 0, x_r, y_r)
x_r, y_r = get_shear_borders(fig0236_a_letter_T, 0, 0.3)
fig0236_a_shear_h = shear_image(fig0236_a_letter_T, 0, 0.3, x_r, y_r)
x_r, y_r = get_shear_borders(fig0237_a_characters_test_pattern, 0.1, 0.4)
fig0237_a_shear = shear_image(fig0237_a_characters_test_pattern, 0.1, 0.4, x_r, y_r)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_shear_v")
ax.imshow(fig0236_a_shear_v, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_shear_h")
ax.imshow(fig0236_a_shear_h, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0237_a_shear")
ax.imshow(fig0237_a_shear, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0237_a_characters_test_pattern")
ax.imshow(fig0237_a_characters_test_pattern, cmap="gray")
"""
Create TranslationImages
"""
fig0236_a_translate_60_20 = translate_image(fig0236_a_letter_T, (60, 20))

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0236_a_translate_60_20")
ax.imshow(fig0236_a_translate_60_20, cmap="gray")

Plotter.save_plots(save_plots, output_dir_AffineTransforms)
Plotter.show_plots(show_plots)

"""
Create IntensityTransforms Images
"""

# Log transform
fig0305_a_dft_log = log_transform(fig0305_a_dft, 1, base=10.0)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0305_a_dft")
ax.imshow(fig0305_a_dft, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0305_a_dft_log")
ax.imshow(fig0305_a_dft_log, cmap="gray")
# gamma transform

fig0308_a_gamma_0_6 = gamma_transform(fig0308_a_fractured_spine, 1, 0.6)
fig0308_a_gamma_0_4 = gamma_transform(fig0308_a_fractured_spine, 1, 0.4)
fig0308_a_gamma_0_3 = gamma_transform(fig0308_a_fractured_spine, 1, 0.3)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0308_a_fractured_spine")
ax.imshow(fig0308_a_fractured_spine, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0308_a_gamma_0_6")
ax.imshow(fig0308_a_gamma_0_6, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0308_a_gamma_0_4")
ax.imshow(fig0308_a_gamma_0_4, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0308_a_gamma_0_3")
ax.imshow(fig0308_a_gamma_0_3, cmap="gray")

fig0309_gamma_3 = gamma_transform(fig0309_a_arial, 1, 3.0)
fig0309_gamma_4 = gamma_transform(fig0309_a_arial, 1, 4.0)
fig0309_gamma_5 = gamma_transform(fig0309_a_arial, 1, 5.0)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0309_a_arial")
ax.imshow(fig0309_a_arial, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0309_gamma_3")
ax.imshow(fig0309_gamma_3, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0309_gamma_4")
ax.imshow(fig0309_gamma_4, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0309_gamma_5")
ax.imshow(fig0309_gamma_5, cmap="gray")

#contrast Strech
max_fig0310 = np.max(fig0310_b_washedout_pollen)
min_fig0310 = np.min(fig0310_b_washedout_pollen)
fig0310_cont_str = contrast_stretching(fig0310_b_washedout_pollen,
                                       (min_fig0310, 0),
                                       (max_fig0310, 255))
fig0310_cont_str_thresh = contrast_stretching(fig0310_b_washedout_pollen,
                                       (110, 0),
                                       (110, 255))
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0310_b_washedout_pollen")
ax.imshow(fig0310_b_washedout_pollen, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0310_cont_str")
ax.imshow(fig0310_cont_str, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0310_cont_str_thresh")
ax.imshow(fig0310_cont_str_thresh, cmap="gray")

# Grey level slicing
fig0312_a_kidney_sliced = intensity_level_slicing_two_tones(fig0312_a_kidney, (150, 255), 50, 200)
fig0312_a_kidney_sliced_linear_0_4 = intensity_level_slicing_scaled(fig0312_a_kidney, (150, 255), 0.6, 180)
fig0312_a_kidney_sliced_linear_0_1 = intensity_level_slicing_scaled(fig0312_a_kidney, (150, 255), 0.3, 200)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0312_a_kidney")
ax.imshow(fig0312_a_kidney, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0312_a_kidney_sliced")
ax.imshow(fig0312_a_kidney_sliced, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0312_a_kidney_sliced_linear_0_4")
ax.imshow(fig0312_a_kidney_sliced_linear_0_4, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0312_a_kidney_sliced_linear_0_1")
ax.imshow(fig0312_a_kidney_sliced_linear_0_1, cmap="gray", vmin=0, vmax=255)

# bit level slicing
fig0314_a_dollar_bill_lvl_0 = bit_level_slicing(fig0314_a_dollar_bill, [0, 0, 0, 0, 0, 0, 0, 1])
fig0314_a_dollar_bill_lvl_1 = bit_level_slicing(fig0314_a_dollar_bill, [0, 0, 0, 0, 0, 0, 1, 0])
fig0314_a_dollar_bill_lvl_2 = bit_level_slicing(fig0314_a_dollar_bill, [0, 0, 0, 0, 0, 1, 0, 0])
fig0314_a_dollar_bill_lvl_3 = bit_level_slicing(fig0314_a_dollar_bill, [0, 0, 0, 0, 1, 0, 0, 0])
fig0314_a_dollar_bill_lvl_4 = bit_level_slicing(fig0314_a_dollar_bill, [0, 0, 0, 1, 0, 0, 0, 0])
fig0314_a_dollar_bill_lvl_5 = bit_level_slicing(fig0314_a_dollar_bill, [0, 0, 1, 0, 0, 0, 0, 0])
fig0314_a_dollar_bill_lvl_6 = bit_level_slicing(fig0314_a_dollar_bill, [0, 1, 0, 0, 0, 0, 0, 0])
fig0314_a_dollar_bill_lvl_7 = bit_level_slicing(fig0314_a_dollar_bill, [1, 0, 0, 0, 0, 0, 0, 0])

fig0314_a_dollar_bill_lvl_7_6 = fig0314_a_dollar_bill_lvl_7 + fig0314_a_dollar_bill_lvl_6
fig0314_a_dollar_bill_lvl_7_6_5 = fig0314_a_dollar_bill_lvl_7_6 + fig0314_a_dollar_bill_lvl_5
fig0314_a_dollar_bill_lvl_7_6_5_4 = fig0314_a_dollar_bill_lvl_7_6_5 + fig0314_a_dollar_bill_lvl_4

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill")
ax.imshow(fig0314_a_dollar_bill, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_0")
ax.imshow(fig0314_a_dollar_bill_lvl_0, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_1")
ax.imshow(fig0314_a_dollar_bill_lvl_1, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_2")
ax.imshow(fig0314_a_dollar_bill_lvl_2, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_3")
ax.imshow(fig0314_a_dollar_bill_lvl_3, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_4")
ax.imshow(fig0314_a_dollar_bill_lvl_4, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_5")
ax.imshow(fig0314_a_dollar_bill_lvl_5, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_6")
ax.imshow(fig0314_a_dollar_bill_lvl_6, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_7")
ax.imshow(fig0314_a_dollar_bill_lvl_7, cmap="gray")


_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_7_6")
ax.imshow(fig0314_a_dollar_bill_lvl_7_6, cmap="gray", vmin=0, vmax=255)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_7_6_5")
ax.imshow(fig0314_a_dollar_bill_lvl_7_6_5, cmap="gray", vmin=0, vmax=255)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0314_a_dollar_bill_lvl_7_6_5_4")
ax.imshow(fig0314_a_dollar_bill_lvl_7_6_5_4, cmap="gray", vmin=0, vmax=255)


Plotter.save_plots(save_plots, output_dir_IntensityTransforms)
Plotter.show_plots(show_plots)

"""
Create Histograms Images
"""

fig0320_b_washedout_pollen_2_global = global_hist_eq(fig0320_b_washedout_pollen_2)
Plotter.create_histogram(fig0320_b_washedout_pollen_2, "fig0320_b_washedout_pollen_2_hist")
Plotter.create_histogram(fig0320_b_washedout_pollen_2_global, "fig0320_b_washedout_pollen_2_global_hist")
Plotter.create_histogram_transformation_func(fig0320_b_washedout_pollen_2, "fig0320_b_washedout_pollen_2_hist_tf")
Plotter.create_histogram_transformation_func(fig0320_b_washedout_pollen_2_global, "fig0320_b_washedout_pollen_2_global_hist_tf")

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0320_b_washedout_pollen_2")
ax.imshow(fig0320_b_washedout_pollen_2, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0320_b_washedout_pollen_2_global")
ax.imshow(fig0320_b_washedout_pollen_2_global, cmap="gray", vmin=0, vmax=255)

fig0326_a_hidden_icons_local = block_hist_eq(fig0326_a_hidden_icons)
fig0326_a_hidden_icons_local_16_40 = block_hist_eq(fig0326_a_hidden_icons, 16)
fig0326_a_hidden_icons_local_16_80 = block_hist_eq(fig0326_a_hidden_icons, 16, 80.0)
fig0326_a_hidden_icons_local_32_160 = block_hist_eq(fig0326_a_hidden_icons, 32, 160.0)
fig0326_a_hidden_icons_local_32_255 = block_hist_eq(fig0326_a_hidden_icons, 32, 255.0)
fig0326_a_hidden_icons_local_32_20 = block_hist_eq(fig0326_a_hidden_icons, 32, 20.0)
fig0326_a_hidden_icons_global = global_hist_eq(fig0326_a_hidden_icons)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0326_a_hidden_icons")
ax.imshow(fig0326_a_hidden_icons, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0326_a_hidden_icons_local")
ax.imshow(fig0326_a_hidden_icons_local, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0326_a_hidden_icons_local_16_40")
ax.imshow(fig0326_a_hidden_icons_local_16_40, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0326_a_hidden_icons_local_16_80")
ax.imshow(fig0326_a_hidden_icons_local_16_80, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0326_a_hidden_icons_local_32_160")
ax.imshow(fig0326_a_hidden_icons_local_32_160, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0326_a_hidden_icons_local_32_255")
ax.imshow(fig0326_a_hidden_icons_local_32_255, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0326_a_hidden_icons_local_32_20")
ax.imshow(fig0326_a_hidden_icons_local_32_20, cmap="gray", vmin=0, vmax=255)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0326_a_hidden_icons_global")
ax.imshow(fig0326_a_hidden_icons_global, cmap="gray", vmin=0, vmax=255)

Plotter.save_plots(save_plots, output_dir_Histograms)
Plotter.show_plots(show_plots)

"""
Create kernel_methods Images
"""

fig0235_d_local_averaging_out = uniform_neighborhood_averaging(fig0235_c_kidney, 41)


_, ax = Plotter.get_zero_padded_fig_and_ax("fig0235_c_local_averaging")
ax.imshow(fig0235_c_kidney, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0235_d_local_averaging_out")
ax.imshow(fig0235_d_local_averaging_out, cmap="gray")


fig0335_a_pcb_noise_avg = uniform_neighborhood_averaging(fig0335_a_pcb_noise, 3)
fig0335_a_pcb_noise_gauss_sxsy_0_4 = gaussian_filtering(fig0335_a_pcb_noise, 3, 0.4, 0.4)
fig0335_a_pcb_noise_max = max_filtering(fig0335_a_pcb_noise, 3)
fig0335_a_pcb_noise_median = median_filtering(fig0335_a_pcb_noise, 3)
fig0335_a_pcb_noise_min = min_filtering(fig0335_a_pcb_noise, 3)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0335_a_pcb_noise")
ax.imshow(fig0335_a_pcb_noise, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0335_a_pcb_noise_avg")
ax.imshow(fig0335_a_pcb_noise_avg, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0335_a_pcb_noise_gauss_sxsy_0_4")
ax.imshow(fig0335_a_pcb_noise_gauss_sxsy_0_4, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0335_a_pcb_noise_max")
ax.imshow(fig0335_a_pcb_noise_max, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0335_a_pcb_noise_median")
ax.imshow(fig0335_a_pcb_noise_median, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0335_a_pcb_noise_min")
ax.imshow(fig0335_a_pcb_noise_min, cmap="gray")


fig0340_a_dipxe_max_5 = max_filtering(fig0340_a_dipxe, 5)
fig0340_a_dipxe_min_5 = min_filtering(fig0340_a_dipxe, 5)
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0340_a_dipxe")
ax.imshow(fig0340_a_dipxe, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0340_a_dipxe_max_5")
ax.imshow(fig0340_a_dipxe_max_5, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0340_a_dipxe_min_5")
ax.imshow(fig0340_a_dipxe_min_5, cmap="gray")


Plotter.save_plots(save_plots, output_dir_kernel_methods)
Plotter.show_plots(show_plots)

"""
Create Operators Images
"""

fig0343_a_skeleton_sobel = Sobel(fig0343_a_skeleton)
fig0343_a_skeleton_sobel_x = Sobel(fig0343_a_skeleton, dx=1, dy=0)
fig0343_a_skeleton_sobel_y = Sobel(fig0343_a_skeleton, dx=0, dy=1)
fig0343_a_skeleton_Laplacian = Laplacian(fig0343_a_skeleton)


_, ax = Plotter.get_zero_padded_fig_and_ax("fig0343_a_skeleton")
ax.imshow(fig0343_a_skeleton, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0343_a_skeleton_sobel")
ax.imshow(fig0343_a_skeleton_sobel, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0343_a_skeleton_sobel_x")
ax.imshow(fig0343_a_skeleton_sobel_x, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0343_a_skeleton_sobel_y")
ax.imshow(fig0343_a_skeleton_sobel_y, cmap="gray")
_, ax = Plotter.get_zero_padded_fig_and_ax("fig0343_a_skeleton_Laplacian")
ax.imshow(fig0343_a_skeleton_Laplacian, cmap="gray")

Plotter.save_plots(save_plots, output_dir_Operators)
Plotter.show_plots(show_plots)
