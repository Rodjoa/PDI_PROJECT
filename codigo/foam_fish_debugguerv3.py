import os
import cv2
import vedo
import json
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
#import PyQt5
#os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
#    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
#)

from time import time
import sys
sys.path.append("/home/mass_estimation")
from BiomassEstimator.FishGeoMetrics.FishGeoMetricsClass import FishGeoMetrics
from BiomassEstimator.PostMaskProcessing.PostMaskProcessingClass import PostMaskProcessing
####################################### CONFIGURATION #########################################
BASE_FOLDER                 = "/home/mass_estimation/pending_videos_output/FoamFishExperiments/"
SVO_FOLDER                  = "torsion_away_from_camera_test"#"torsion_away_from_camera_test"
FRAMES                      = [301] #Keep empty to evaluate all frames in target folder
ACTIVATE_VISUALIZATION      = True
ACTIVATE_ROTATION_VIS       = False
PLOT_TRAJECTORY             = False
PLOT_LENGTHS                = True
GROUND_THRUTH_VALUE         = 27.5
#----------------------------------------------------------------------------------------------
SAVE_TO_RESULTS             = False
RESULTS_FILENAME            = SVO_FOLDER + ".csv"
###############################################################################################


def analise_samples(base_folder, obj_folder, frames, vis=False):
    input_data_path = "{}{}/".format(base_folder, obj_folder)
    with open(input_data_path + 'cam_intrinsic_params.json', 'r') as file:
        params_dict = json.load(file)
    
    post_mask_processing = PostMaskProcessing()
    GeoMetrics_instance = FishGeoMetrics(debbug_mode=vis, visualize_rotation=ACTIVATE_ROTATION_VIS)
    GeoMetrics_instance.load_camera_instrinsic_parameters(params_dict)

    sorted_masks_list = sorted(os.listdir(input_data_path + "data_mask"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    executions_times = []
    length_list = []
    used_frames_list = []
    positions_list_dmap = []
    positions_list_world = []
    x_angles = []
    y_angles = []
    z_angles = []
    mask_data_list = []

    for idx, frame_filename in enumerate(sorted_masks_list):
        frame_number = frame_filename.split('_')[-1].split('.')[0]
        if len(frames) != 0 and int(frame_number) not in frames:
            continue
        print("\nFrame: {}".format(frame_number))

        # Carga datos de entrada
        fish_frame, depth_map, pcd = get_frame_data(input_data_path, frame_number)
        fish_mask = cv2.imread(input_data_path + "/data_mask/{}.png".format(frame_number), cv2.IMREAD_GRAYSCALE)
        bbox = get_bbox(fish_mask)
        post_mask_processing.set_bbox(bbox)

        # 1. Preprocesamiento mejorado de la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fish_mask = cv2.morphologyEx(fish_mask, cv2.MORPH_CLOSE, kernel)  # Cerrar huecos
        fish_mask = cv2.GaussianBlur(fish_mask, (3, 3), 0)  # Suavizar bordes
        edges = cv2.Canny(fish_mask, 50, 150)  # Detectar bordes

        # 2. Validación con mapa de profundidad
        depth_map_filtered = cv2.bilateralFilter(depth_map, 10, 70, 70)  # Suavizar profundidad sin perder bordes
        gradient_x = cv2.Sobel(depth_map_filtered, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(depth_map_filtered, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        fish_mask[magnitude < 10] = 0  # Elimina regiones de bajo gradiente

        # Procesar máscara y verificar
        fish_mask, message = post_mask_processing.handle_mask_depth_data(fish_mask, depth_map, proportion=0.35)
        if fish_mask is None:
            print("Unviable sample: {}".format(message))
            continue

        # 3. Suavizado y validación de contornos
        contours, _ = cv2.findContours(fish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon=0.01 * cv2.arcLength(largest_contour, True), closed=True)
            fish_mask[:] = 0
            cv2.drawContours(fish_mask, [smoothed_contour], -1, 255, thickness=-1)

        bgr_fish_mask = post_mask_processing.get_bgr_mask()
        mask_data_distribution = get_bgr_mask_color_proportion(bgr_fish_mask)
        mask_data_list.append(mask_data_distribution)

        # Calcular longitud y altura
        start = time()
        length, height = GeoMetrics_instance.get_fish_length_and_height(
            frame=fish_frame,
            depth_map=depth_map,
            pcd=pcd,
            treated_segmentation_mask=fish_mask,
            bgr_mask=bgr_fish_mask
        )
        end = time()
        exe_time = end - start
        print("\tEstimated length: {:.2f} cm".format(length))
        print("\tEstimation time : {:.2f} s".format(exe_time))
        executions_times.append(exe_time)
        length_list.append(length)
        used_frames_list.append(int(frame_number))

        # Obtener datos de posición
        x_correction_angle, y_correction_angle, z_correction_angle, initial_position_dmap, initial_position_world = GeoMetrics_instance.get_initial_position_data()
        positions_list_dmap.append(initial_position_dmap)
        positions_list_world.append(initial_position_world)
        x_angles.append(np.degrees(x_correction_angle))
        y_angles.append(np.degrees(y_correction_angle))
        z_angles.append(np.degrees(z_correction_angle))

    # Estadísticas
    print("length mean: {:.2f} cm".format(np.mean(length_list)))
    print("length std: {:.2f} cm".format(np.std(length_list)))
    print("mean execution time: {:.2f}".format(np.mean(executions_times)))

    # Graficar resultados
    if PLOT_LENGTHS:
    # Calcular el MSE
        mse = np.mean([(x - GROUND_THRUTH_VALUE)**2 for x in length_list])
        print("Mean Squared Error (MSE): {:.2f}".format(mse))

        # Graficar las longitudes
        plt.plot(used_frames_list, length_list, '-o', label='mean: {:.2f}, std: {:.2f}'.format(np.mean(length_list), np.std(length_list)))
        plt.xlabel('Frame number')
        plt.tick_params(axis='x', rotation=45)
        plt.ylabel('Estimated length [cm]')
        plt.legend()
        plt.hlines(y=GROUND_THRUTH_VALUE, xmin=min(used_frames_list), xmax=max(used_frames_list), colors='r', label='GT')
        plt.title('Video: {} | Length estimations\nMSE: {:.2f}'.format(SVO_FOLDER, mse))  # Mostrar el MSE en el título
        plt.show()

        plot_fish_angle(used_frames_list, x_angles, y_angles, z_angles)


    if PLOT_TRAJECTORY:
        plot_trajectory(used_frames_list, positions_list_dmap, positions_list_world)
    
    if SAVE_TO_RESULTS:
        save_results_as_csv(RESULTS_FILENAME, used_frames_list, length_list, x_angles, y_angles, z_angles, positions_list_dmap, positions_list_world, mask_data_list)


def show_high_std_samples(base_folder, obj_folder):
    fish_estimations = pd.read_csv(base_folder + obj_folder + "/estimations_per_fish.csv")
    df_sorted = fish_estimations.sort_values(by='Length std', ascending=False)
    print("All samples")
    print(df_sorted.head())

    fish_estimations = pd.read_csv(base_folder + obj_folder + "/not_occluded_estimations_per_fish.csv")
    df_sorted = fish_estimations.sort_values(by='Length std', ascending=False)
    print("Not occluded samples")
    print(df_sorted.head())

def filter_bgr_mask(bgr_mask):
    #RETURN ONLY WHITE PORTION
    white_pixels_mask = np.all(bgr_mask == [255, 255, 255], axis=-1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_pixels_mask.astype(np.uint8))
    largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_segment_mask = (labels == largest_component)
    
    filtered_bgr_mask = np.copy(bgr_mask)
    filtered_bgr_mask[~largest_segment_mask & white_pixels_mask] = [0, 255, 0]

    #cv2.imshow("filtered_bgr_mask", filtered_bgr_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return



def get_bgr_mask_color_proportion(bgr_mask):
    white_pixels_mask = np.all(bgr_mask == [255, 255, 255], axis=-1)
    total_white_pixels = np.sum(white_pixels_mask)
    
    blue_pixels_mask  = np.all(bgr_mask == [255,   0,   0], axis=-1)
    total_blue_pixels = np.sum(blue_pixels_mask)

    red_pixels_mask   = np.all(bgr_mask == [  0,   0, 255], axis=-1)
    total_red_pixels = np.sum(red_pixels_mask)
    
    total_mask_pixels = total_white_pixels + total_blue_pixels + total_red_pixels
    
    white_proportion = 100*total_white_pixels/total_mask_pixels
    blue_proportion  = 100*total_blue_pixels/total_mask_pixels
    red_proportion   = 100*total_red_pixels/total_mask_pixels
    
    print("\tWhite proportion : {:.2f} %".format(white_proportion))
    print("\tBlue  proportion : {:.2f} %".format(blue_proportion))
    print("\tRed   proportion : {:.2f} %".format(red_proportion))
    return (total_mask_pixels, white_proportion, blue_proportion, red_proportion)

def get_frame_data(root_folder_path, frame_to_visualize):
    fish_frame_path = root_folder_path + "/data_img/{}.png".format(frame_to_visualize)
    fish_frame = cv2.imread(fish_frame_path, cv2.IMREAD_COLOR)
    
    depth_map_path = os.path.join(root_folder_path + "/data_depth/{}.npy".format(frame_to_visualize))
    depth_map = np.load(depth_map_path)

    pcd_path = os.path.join(root_folder_path + "/data_cloud/{}.npy".format(frame_to_visualize))
    pcd = np.load(pcd_path)
    
    return fish_frame, depth_map, pcd

def get_bbox(mask):
    '''
    Receives 0-255 binary mask and returns the respective bbox
    in format (x1, y1, x2, y2)
    '''
    # Find the non-zero pixels in the mask
    y_indices, x_indices = np.where(mask > 0)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # Return None if the mask is empty
    
    # Get the bounding box coordinates
    x1, x2 = np.min(x_indices), np.max(x_indices)
    y1, y2 = np.min(y_indices), np.max(y_indices)
    
    # Return the bounding box in format (x1, y1, x2, y2)
    return (x1, y1, x2, y2)

def save_results_as_csv(filename, frame_list, length_list, x_angle_list, y_angle_list, z_angle_list, pos_in_dmap, pos_in_world, mask_data_list):
    save_path = "/home/mass_estimation/BiomassEstimator/Utils/FoamFishDebugguer/results/"
    data = zip(frame_list, length_list, x_angle_list, y_angle_list, z_angle_list, pos_in_dmap, pos_in_world, mask_data_list)
    df = pd.DataFrame(columns=['Frame', 'Length', 'x_angle', 'y_angle', 'z_angle', 'position_in_dmap', 'position_in_world', 'mask_proportion'])
    for frame, length, x_angle, y_angle, z_angle, pos_dmap, pos_world, mask_data in data:
        df.loc[len(df)] = [frame, length, x_angle, y_angle, z_angle, pos_dmap, pos_world, mask_data]
    df.to_csv(save_path + filename, index=False)

def plot_trajectory(frame_list, positions_dmap, positions_world):
    x_values = [x for (x, y, z) in positions_dmap]
    y_values = [y for (x, y, z) in positions_dmap]
    z_values = [z for (x, y, z) in positions_dmap]
    distance_to_camera = [np.sqrt(x**2 + y**2 + z**2) for (x, y, z) in positions_world]
    #for (x, y, z) in positions_world:
    #    print("coords: {:.2f}, {:.2f}, {:.2f}".format(x, y, z))
    # Create a figure and two subplots (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the pixel positions in the first subplot
    ax[0].plot(x_values, y_values, '-o')
    ax[0].scatter(x_values[0], y_values[0], color='green', label='Start', zorder=5)
    ax[0].scatter(x_values[-1], y_values[-1], color='red', label='End', zorder=5)
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].set_title('Frame position')
    ax[0].set_xlim(0, 1920)
    ax[0].set_ylim(1080, 0)
    ax[0].legend()

    # Plot the distance to camera in the second subplot
    ax[1].plot(frame_list, distance_to_camera, '-o')
    ax[1].set_xlabel('Frame number')
    ax[1].set_ylabel('Raw distance [mm]')
    ax[1].set_title('Distance to camera per frame')
    ax[1].tick_params(axis='x', rotation=45)
    

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()

def plot_fish_angle(frame_list, x_angles, y_angles, z_angles):
    plt.plot(frame_list, x_angles,  '-o', label='X', color='blue')
    plt.plot(frame_list, y_angles,  '-', label='Y', color='black')
    plt.plot(frame_list, z_angles,  '--', label='Z', color='purple')
    plt.xlabel('Frame number')
    plt.tick_params(axis='x', rotation=45)
    plt.ylabel('Rotation angle')
    #plt.ylim(-180, 180)
    plt.legend()
    plt.title('Video: {}| Rotation'.format(SVO_FOLDER))
    plt.show()

if __name__ == "__main__":
    #show_high_std_samples(base_folder = BASE_FOLDER,
    #                obj_folder  = SVO_FOLDER)
    
    analise_samples(base_folder = BASE_FOLDER,
                    obj_folder  = SVO_FOLDER,
                    frames      = FRAMES,
                    vis         = ACTIVATE_VISUALIZATION)