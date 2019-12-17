import numpy as np
import cv2
from plotTools import Plotter

Fig0431_d_Location = "../Datasets/DIP3E_Original_Images_CH04/Fig0431(d)(blown_ic_crop).tif"
show_images = True
save_images = True
save_directory = "out/Question_6/"
twiddle_factor = 10**-30


def gaussian_filter_function(u, v, mu_u, mu_v, sigma_sq):
    d_sq_uv = (u-mu_u)**2+(v-mu_v)**2
    d_sq_uv_div_sig = d_sq_uv/(2.0*sigma_sq)
    return np.e**-d_sq_uv_div_sig


def get_gaussian_window(size_u, size_v, sigma_sq):
    u_points, v_points = np.meshgrid(np.arange(size_u), np.arange(size_v))
    return gaussian_filter_function(u_points, v_points, (size_u-1)/2.0, (size_v-1)/2.0, sigma_sq)


fig4_36_a = cv2.imread(Fig0431_d_Location, cv2.IMREAD_GRAYSCALE)

fig4_36_b = np.zeros(np.array(fig4_36_a.shape)*2)
fig4_36_b[:fig4_36_a.shape[0], :fig4_36_a.shape[1]] = fig4_36_a[:, :]

fig4_36_b_fft = np.fft.fft2(fig4_36_b)
fig4_36_d = np.fft.fftshift(fig4_36_b_fft)
fig4_36_c = np.real(np.fft.ifft2(fig4_36_d))

fig4_36_e = get_gaussian_window(fig4_36_d.shape[1], fig4_36_d.shape[0], 320)
fig4_36_f = fig4_36_d*fig4_36_e
fig4_36_g = np.real(np.fft.ifft2(np.fft.fftshift(fig4_36_f)))
fig4_36_h = fig4_36_g[:fig4_36_a.shape[0], :fig4_36_a.shape[1]]

fig4_36_a_out, ax = Plotter.get_zero_padded_fig_and_ax("fig4_36_a")
ax.imshow(fig4_36_a, cmap='gray', interpolation='nearest')

fig4_36_b_out, ax = Plotter.get_zero_padded_fig_and_ax("fig4_36_b")
ax.imshow(fig4_36_b, cmap='gray', interpolation='nearest')

fig4_36_c_out, ax = Plotter.get_zero_padded_fig_and_ax("fig4_36_c")
ax.imshow(fig4_36_c, cmap='gray', interpolation='nearest', vmin=0)

fig4_36_d_out, ax = Plotter.get_zero_padded_fig_and_ax("fig4_36_d")
ax.imshow(20*np.log10(np.abs(fig4_36_d)+twiddle_factor), cmap='gray', interpolation='nearest', vmin=55)

fig4_36_e_out, ax = Plotter.get_zero_padded_fig_and_ax("fig4_36_e")
ax.imshow(fig4_36_e, cmap='gray', interpolation='nearest')

fig4_36_f_out, ax = Plotter.get_zero_padded_fig_and_ax("fig4_36_f")
ax.imshow(20*np.log10(np.abs(fig4_36_f)+twiddle_factor), cmap='gray', interpolation='nearest', vmin=55)

fig4_36_g_out, ax = Plotter.get_zero_padded_fig_and_ax("fig4_36_g")
ax.imshow(fig4_36_g, cmap='gray', interpolation='nearest')

fig4_36_h_out, ax = Plotter.get_zero_padded_fig_and_ax("fig4_36_h")
ax.imshow(fig4_36_h, cmap='gray', interpolation='nearest')

Plotter.save_plots(save_images, save_directory)
Plotter.show_plots(show_images)

