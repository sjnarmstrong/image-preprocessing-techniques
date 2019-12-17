
from BuildingBlocks.Histograms import global_hist_eq

from plotTools import Plotter
import cv2


show_plots = False
save_plots = True

output_dir_AffineTransforms = "out/Building_blocks/AffineTransforms/"
output_dir_Histograms = "out/Building_blocks/Histograms/"
output_dir_IntensityTransforms = "out/Building_blocks/IntensityTransforms/"
output_dir_kernel_methods = "out/Building_blocks/kernel_methods/"
output_dir_Operators = "out/Building_blocks/Operators/"



Fig0320_b_Location = "../Datasets/DIP3E_Original_Images_CH03/Fig0320(2)(2nd_from_top).tif"


def safe_read_image(location):
    hold_out = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
    if hold_out is None:
        print("Could not find image at: " + location)
        print("Please ensure that you have extracted the datasets to the relevant locations")
        exit(0)
    return hold_out


fig0320_b_washedout_pollen_2 = safe_read_image(Fig0320_b_Location)

fig0320_b_washedout_pollen_2_global = global_hist_eq(fig0320_b_washedout_pollen_2)
Plotter.create_histogram(fig0320_b_washedout_pollen_2, "fig0320_b_washedout_pollen_2_hist")
Plotter.create_histogram(fig0320_b_washedout_pollen_2_global, "fig0320_b_washedout_pollen_2_global_hist")
Plotter.create_histogram_transformation_func(fig0320_b_washedout_pollen_2, "fig0320_b_washedout_pollen_2_hist_tf")
Plotter.create_histogram_transformation_func(fig0320_b_washedout_pollen_2_global, "fig0320_b_washedout_pollen_2_global_hist_tf")


output_dir_Histograms = "out/Building_blocks/Histograms/"
Plotter.save_plots(save_plots, output_dir_Histograms)