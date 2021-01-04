import matplotlib.pyplot as plt
import numpy as np
import gdal
import os

all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
                 "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
rasterized_shapefiles_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\District_Shapefiles_as_Clipping_bands\\"
unclipped_images_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\30m_4326_btt_2020_unclipped_images\\"


def visualize_rasterized_shapefiles():
    global all_districts, rasterized_shapefiles_path
    for district in all_districts:
        this_shapefile_path = os.path.join(rasterized_shapefiles_path, f"{district}_shapefile.tif")
        ds = gdal.Open(this_shapefile_path)
        assert ds.RasterCount == 1
        shapefile_mask = ds.GetRasterBand(1).ReadAsArray()
        plt.title(f"District: {district}; Shape: {shapefile_mask.shape}")
        plt.imshow(shapefile_mask)
        plt.show()
    pass


def mask_landsat8_using_rasterized_shapefiles():
    global all_districts, rasterized_shapefiles_path
    for district in all_districts:
        this_shapefile_path = os.path.join(rasterized_shapefiles_path, f"{district}_shapefile.tif")
        ds = gdal.Open(this_shapefile_path)
        assert ds.RasterCount == 1
        shapefile_mask = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)
        image_ds = gdal.Open(os.path.join(unclipped_images_path, f"landsat8_4326_30_2015_region_{district}.tif"))
        clipped_full_spectrum = list()
        print("{}: Shapefile Size: {}".format(district, shapefile_mask.shape))
        for x in range(1, image_ds.RasterCount + 1):
            this_band = image_ds.GetRasterBand(x).ReadAsArray()
            print("{}: Band-{} Size: {}".format(district, x, this_band.shape))
            clipped_full_spectrum.append(np.multiply(this_band, shapefile_mask))
        x_prev, y_prev = clipped_full_spectrum[0].shape
        x_fixed, y_fixed = int(128 * np.ceil(x_prev / 128)), int(128 * np.ceil(y_prev / 128))
        diff_x, diff_y = x_fixed - x_prev, y_fixed - y_prev
        diff_x_before, diff_y_before = diff_x//2, diff_y//2
        clipped_full_spectrum_resized = [np.pad(x, [(diff_x_before, diff_x-diff_x_before), (diff_y_before, diff_y-diff_y_before)], mode='constant')
                                         for x in clipped_full_spectrum]
        clipped_full_spectrum_stacked_image = np.dstack(clipped_full_spectrum_resized)
        plt.title(f"District: {district}; Shape: {clipped_full_spectrum_stacked_image.shape}")
        plt.imshow(clipped_full_spectrum_stacked_image[:,:,[3,2,1]])
        plt.show()
    pass


if __name__ == "__main__":
    # visualize_rasterized_shapefiles()
    mask_landsat8_using_rasterized_shapefiles()
