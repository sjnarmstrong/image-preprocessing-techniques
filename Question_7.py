import numpy as np
import cv2
from plotTools import Plotter

Fig0438_a_Location = "../Datasets/DIP3E_Original_Images_CH04/Fig0438(a)(bld_600by600).tif"
show_images = True
save_images = True
save_directory = "out/Question_7/"
twiddle_factor = 10**-30

spacial_mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

fig0438_a = cv2.imread(Fig0438_a_Location, cv2.IMREAD_GRAYSCALE)
fig0438_a_padded = np.pad(fig0438_a,
                          ((0, fig0438_a.shape[0]),
                           (0, fig0438_a.shape[1])),
                          'constant')
fig0438_b = np.fft.fftshift(np.fft.fft2(fig0438_a_padded))


spacial_mask_padded = np.pad(spacial_mask,
                             ((0, fig0438_a_padded.shape[0]-spacial_mask.shape[0]),
                              (0, fig0438_a_padded.shape[1]-spacial_mask.shape[1])),
                             'constant')

fft_filter = np.fft.fft2(-spacial_mask_padded)
fft_filter_shifted = np.fft.fftshift(fft_filter)

fft_filter_mag = 20*np.log10(np.abs(fft_filter_shifted)+1)

fig0438_b_fft_filtered = fft_filter_shifted*fig0438_b
fig0438_b_filtered = np.real(np.fft.ifft2(np.fft.fftshift(fig0438_b_fft_filtered)))[:fig0438_a.shape[0],
                                                                                    :fig0438_a.shape[1]]

fig0438_a_filtered = cv2.filter2D(fig0438_a, cv2.CV_64F, spacial_mask)


_, ax = Plotter.get_zero_padded_fig_and_ax("fft_filter_mag")
ax.imshow(fft_filter_mag, cmap='gray')

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0438_a")
ax.imshow(fig0438_a, cmap='gray')

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0438_b")
ax.imshow(20*np.log10(np.abs(fig0438_b)+twiddle_factor), cmap='gray', vmin=40)

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0438_b_filtered")
ax.imshow(fig0438_b_filtered, cmap='gray')

_, ax = Plotter.get_zero_padded_fig_and_ax("fig0438_a_filtered")
ax.imshow(fig0438_a_filtered, cmap='gray')

Plotter.save_plots(save_images, save_directory)
Plotter.show_plots(show_images)
