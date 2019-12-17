import numpy as np
import cv2

from plotTools import Plotter

Fig0222_a_Location = "../Datasets/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif"
Fig0222_b_Location = "../Datasets/DIP3E_Original_Images_CH02/Fig0222(b)(cameraman).tif"
Fig0222_c_Location = "../Datasets/DIP3E_Original_Images_CH02/Fig0222(c)(crowd).tif"
show_images = True
save_images = True
save_directory = "out/Question_8/"
twiddle_factor = 10**-30


def lowpass_butterworth_filter(u, v, mu_u, mu_v, n, D_0):
    d_sq_uv = (((u-mu_u)**2+(v-mu_v)**2)/(D_0**2))**n
    return 1.0/(1+d_sq_uv)


def highpass_butterworth_filter(u, v, mu_u, mu_v, n, D_0):
    return 1.0-lowpass_butterworth_filter(u, v, mu_u, mu_v, n, D_0)


def get_window(size_u, size_v, n, D_0, filter_type=lowpass_butterworth_filter):
    u_points, v_points = np.meshgrid(np.arange(size_u), np.arange(size_v))
    return filter_type(u_points, v_points, (size_u-1)/2.0, (size_v-1)/2.0, n, D_0)


def filter_image(img):
    img_padded = np.pad(img,
                        ((0, img.shape[0]),
                         (0, img.shape[1])),
                        'constant')
    img_fft = np.fft.fftshift(np.fft.fft2(img_padded))
    lowpass_filter = get_window(img_padded.shape[0], img_padded.shape[1], 2, 20)
    highpass_filter = get_window(img_padded.shape[0], img_padded.shape[1], 2, 20, highpass_butterworth_filter)
    img_fft_low = lowpass_filter*img_fft
    img_fft_high = highpass_filter*img_fft
    return np.real(np.fft.ifft2(np.fft.fftshift(img_fft_low)))[:img.shape[0], :img.shape[1]],\
           np.real(np.fft.ifft2(np.fft.fftshift(img_fft_high)))[:img.shape[0], :img.shape[1]]


#lowpass_filter = get_window(128, 128, 2, 20)
#lowpass_filter_spatial = np.real(np.fft.ifft2(np.fft.fftshift(lowpass_filter)))
#highpass_filter = get_window(128, 128, 2, 20, highpass_butterworth_filter)

fig2_22_a = cv2.imread(Fig0222_a_Location, cv2.IMREAD_GRAYSCALE)
fig2_22_b = cv2.imread(Fig0222_b_Location, cv2.IMREAD_GRAYSCALE)
fig2_22_c = cv2.imread(Fig0222_c_Location, cv2.IMREAD_GRAYSCALE)


fig2_22_a_low, fig2_22_a_high = filter_image(fig2_22_a)
fig2_22_b_low, fig2_22_b_high = filter_image(fig2_22_b)
fig2_22_c_low, fig2_22_c_high = filter_image(fig2_22_c)


_, ax = Plotter.get_zero_padded_fig_and_ax("fig2_22_a_high")
ax.imshow(fig2_22_a_high, cmap='gray')
_, ax = Plotter.get_zero_padded_fig_and_ax("fig2_22_a_low")
ax.imshow(fig2_22_a_low, cmap='gray')


_, ax = Plotter.get_zero_padded_fig_and_ax("fig2_22_b_high")
ax.imshow(fig2_22_b_high, cmap='gray')
_, ax = Plotter.get_zero_padded_fig_and_ax("fig2_22_b_low")
ax.imshow(fig2_22_b_low, cmap='gray')


_, ax = Plotter.get_zero_padded_fig_and_ax("fig2_22_c_high")
ax.imshow(fig2_22_c_high, cmap='gray')
_, ax = Plotter.get_zero_padded_fig_and_ax("fig2_22_c_low")
ax.imshow(fig2_22_c_low, cmap='gray')


Plotter.save_plots(save_images, save_directory)
Plotter.show_plots(show_images)
