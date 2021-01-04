from plotly.subplots import make_subplots
from skimage.transform import resize
import matplotlib.image as matimg
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import statistics
import os

NULL_PIXEL = 0
NON_FOREST_PIXEL = 1
FOREST_PIXEL = 2
BTT_Forest_Percentages = {
    "abbottabad":   {2015: 39.25},
    "battagram":    {2015: 33.28},
    "buner":        {2015: 32.84},
    "chitral":      {2015: 9.71},
    "hangu":        {2015: 6.56},
    "haripur":      {2015: 30.64},
    "karak":        {2015: 14.85},
    "kohat":        {2015: 20.03},
    "kohistan":     {2015: 37.74},
    "lower_dir":    {2015: 20.09},
    "malakand":     {2015: 15.90},
    "mansehra":     {2015: 31.21},
    "nowshehra":    {2015: 12.20},
    "shangla":      {2015: 39.74},
    "swat":         {2015: 24.64},
    "tor_ghar":     {2015: 29.93},
    "upper_dir":    {2015: 21.08}
}


def decipher_this_array(this_path):
    global NULL_PIXEL, NON_FOREST_PIXEL, FOREST_PIXEL
    this_map = np.load(this_path)
    forest_pixel_count = (this_map == FOREST_PIXEL).sum()
    non_forest_pixel_count = (this_map == NON_FOREST_PIXEL).sum()
    return forest_pixel_count*100/(forest_pixel_count+non_forest_pixel_count), this_map


if __name__ == "__main__":
    do_work, save_png_image, show_image, show_forest_change_trend = True, False, False, False 
    all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
                     "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
    all_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    np_arrays_path = "E:\\Forest Cover - Redo 2020\\BTT_2014_2020_inferences\\rgb\\"
    if do_work:
        for year in all_years:
            # if year == 2015:
            #     print("(LOG): Skipping Reference Year 2015. Maps Already Exist")
            #     continue
            for district in all_districts:
                forest_percentage, forest_map = decipher_this_array(this_path=os.path.join(np_arrays_path, "generated_map_{}_{}.npy".format(district, year)))
                # print("District: {}; Year: {}; Forest Percentage: {:.2f}%".format(district, year, forest_percentage))
                print("District: {}; Year: {}; Size: {}; Forest Percentage: {:.2f}%".format(district, year, forest_map.shape, forest_percentage))
                # BTT_Forest_Percentages[district][year] = forest_percentage
                # if save_png_image:
                #     forest_map_rband = np.zeros_like(forest_map)
                #     forest_map_gband = np.zeros_like(forest_map)
                #     forest_map_bband = np.zeros_like(forest_map)
                #     forest_map_rband[forest_map == 1] = 255
                #     forest_map_gband[forest_map == 2] = 255
                #     forest_map_for_visualization = np.dstack([forest_map_rband, forest_map_gband, forest_map_bband]).astype(np.uint8)
                #     matimg.imsave(f'Forest_Maps/{district}_{year}.png', forest_map_for_visualization)
                # if show_image:
                #     plt.imshow(forest_map)
                #     plt.title(f"District: {district}; Year: {year}")
                #     plt.show()
                # pass
        exit()
    BTT_Forest_Percentages = {
        'abbottabad': {2015: 39.25, 2014: 46.04621722763262, 2016: 34.35884537765893, 2017: 40.26675445473193, 2018: 35.13535897214782, 2019: 34.48254805226638,
                       2020: 45.39154402969262},
        'battagram': {2015: 33.28, 2014: 57.09865877656976, 2016: 55.85722843439331, 2017: 41.95343253580502, 2018: 43.777372272966176, 2019: 54.45379145993235,
                      2020: 58.380243113741024},
        'buner': {2015: 32.84, 2014: 17.26300896354801, 2016: 29.231791102749753, 2017: 35.4449517963355, 2018: 28.76672609821667, 2019: 27.80067721811203,
                  2020: 33.17616036782586},
        'chitral': {2015: 9.71, 2014: 55.6107822834034, 2016: 50.119394331037654, 2017: 51.83242597558477, 2018: 52.98149616109631, 2019: 54.66158620175353,
                    2020: 57.39770756962325},
        'hangu': {2015: 6.56, 2014: 6.974258488881502, 2016: 7.694414873113017, 2017: 3.918326008005088, 2018: 4.466802938798359, 2019: 6.949675854568955,
                  2020: 10.806951650275925},
        'haripur': {2015: 30.64, 2014: 11.622037470625317, 2016: 14.045016062089287, 2017: 24.831748893015394, 2018: 24.39622900328397, 2019: 17.372427222717754,
                    2020: 18.862685608740353},
        'karak': {2015: 14.85, 2014: 5.1322273395450795, 2016: 5.372249972741743, 2017: 8.382502443708383, 2018: 1.6004913877676668, 2019: 4.691228108532314,
                  2020: 10.655032864273872},
        'kohat': {2015: 20.03, 2014: 5.712544164544983, 2016: 5.146213435220055, 2017: 5.597545204878517, 2018: 4.223641387655494, 2019: 5.840710300210809,
                  2020: 13.866801084377316},
        'kohistan': {2015: 37.74, 2014: 46.16415897095371, 2016: 45.373097194008075, 2017: 42.56137158043736, 2018: 43.314186833884776, 2019: 52.84831367215208,
                     2020: 50.002575617977406},
        'lower_dir': {2015: 20.09, 2014: 19.221942788951225, 2016: 14.085914504976683, 2017: 35.713170232335244, 2018: 24.577862824938293,
                      2019: 11.257430491432066, 2020: 17.37342613008638},
        'malakand': {2015: 15.9, 2014: 7.608936548793886, 2016: 17.42199734590376, 2017: 23.57523539520261, 2018: 17.985871206963704, 2019: 8.780410975058627,
                     2020: 11.44416045365352},
        'mansehra': {2015: 31.21, 2014: 41.26736950826714, 2016: 36.785591587542456, 2017: 31.27685388229489, 2018: 33.68184719646843, 2019: 34.44141303302365,
                     2020: 45.19694530990319},
        'nowshehra': {2015: 12.2, 2014: 8.189055212326927, 2016: 5.63672735560001, 2017: 15.527158630428323, 2018: 12.926701435727358, 2019: 9.38773941509468,
                      2020: 14.276708484207965},
        'shangla': {2015: 39.74, 2014: 45.321252987230345, 2016: 56.885720339778615, 2017: 55.714655085474945, 2018: 54.22822932369145, 2019: 56.70803765068885,
                    2020: 64.64124763094637},
        'swat': {2015: 24.64, 2014: 30.482414609611325, 2016: 30.607631873988993, 2017: 36.47893737282138, 2018: 32.72159128599604, 2019: 38.95196240425682,
                 2020: 43.69057734380537},
        'tor_ghar': {2015: 29.93, 2014: 37.709131412561504, 2016: 41.8916141743842, 2017: 51.17070385776038, 2018: 48.26797678992598, 2019: 49.57174862647897,
                     2020: 44.258303715090456},
        'upper_dir': {2015: 21.08, 2014: 52.93012061118925, 2016: 45.509868476204545, 2017: 69.20630857391686, 2018: 61.97824359293754, 2019: 54.815596643640404,
                      2020: 69.40562591776437}}
    if show_forest_change_trend:
        fig = go.Figure()
        for district in all_districts:
            y = np.array([BTT_Forest_Percentages[district][year] for year in all_years])
            fig.add_trace(go.Scatter(x=all_years, y=y, mode='lines+markers', name=f'{district}'))
        fig.add_trace(go.Scatter(x=all_years, y=[statistics.mean([BTT_Forest_Percentages[district][year] for district in all_districts]) for year in all_years],
                                 name='Average Trend', line=dict(color='royalblue', width=4, dash='dash')))
        fig.update_layout(hovermode="x")
        fig.show()
    images_path = 'E:\\Forest Cover - Redo 2020\\Digitized_Forest_Maps_2014_2020_png\\'
    row_count, col_count = 0, 0
    for district in all_districts:
        print(f"Adding District: {district}")
        fig, axs = plt.subplots(1, len(all_years))
        fig.suptitle(f'{district}', fontsize=16)
        col_count = 0
        for year in all_years:
            image = io.imread(os.path.join(images_path, f'{district}_{year}.png'))
            axs[col_count].imshow(image)
            axs[col_count].set_title("{} @ {:.2f}%".format(year, BTT_Forest_Percentages[district][year]))
            axs[col_count].axis('off')
            col_count += 1
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
    pass
