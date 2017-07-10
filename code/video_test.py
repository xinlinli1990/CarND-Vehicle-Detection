import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from find_cars_test import find_cars, update_car_box_lists, draw_labeled_bboxes, add_heat, apply_threshold
from scipy.ndimage.measurements import label

import pickle


def process_image(img):
    global spatial_features_param
    global hist_features_param
    global hog_features_param
    global search_windows
    global clf
    global X_scaler
    global box_lists
    global time_average_param
    averaged_time_steps = time_average_param['averaged_time_steps']
    heat_map_threshold = time_average_param['heat_map_threshold']
    color_space = 'RGB2YCrCb' #YCrCb #HSV

    box_lists = update_car_box_lists(img, search_windows, color_space, spatial_features_param, hist_features_param, hog_features_param, clf, X_scaler, box_lists,
                                     averaged_time_steps)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    for box_list in box_lists:
        heat = add_heat(heat, box_list)
        heat[heat <= 3] = 0

    draw_heat = np.copy(heat)
    heat = apply_threshold(heat, heat_map_threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    #return heatmap

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    # draw_heat = cv2.resize(draw_heat, img.shape[0:2][::-1])
    #
    # draw_heat_R = np.zeros_like(draw_heat)
    # draw_heat_G = np.zeros_like(draw_heat)
    # draw_heat_B = np.zeros_like(draw_heat)
    #
    # draw_heat_R = np.copy(draw_heat)
    # draw_heat_G = np.copy(draw_heat)
    # draw_heat_B = np.copy(draw_heat)
    #
    # draw_heat_R[draw_heat < 260] = 0
    # draw_heat_G[(draw_heat <= 160) | (draw_heat >= 300)] = 0
    # draw_heat_B[draw_heat > 200] = 0
    #
    # draw_img = 1.0 * draw_img + 1.0 * np.asarray(
    #     np.dstack(
    #         (draw_heat_R * 5.0 / averaged_time_steps,
    #          draw_heat_G * 5.0 / averaged_time_steps,
    #          draw_heat_B * 5.0 / averaged_time_steps)))  # np.zeros_like(draw_heat) # , dtype=np.uint8

    draw_img = np.clip(draw_img, 0, 255)

    return draw_img


f_paths = [#'../test_video.mp4',
           '../project_video.mp4',
           #'./cut2.mp4'
           ]
output_paths = [#'./test_video_final.mp4',
                './project_video_win6503.mp4',
                ]

hog_features_param = {'orient': 9,
                      'pix_per_cell': 8,
                      'cell_per_block': 2}

hist_features_param = {'hist_bins': 0}
spatial_features_param = {'spatial_size': (0, 0)}

search_windows = [{'y_start': 400, 'y_stop': 700},
                  {'y_start': 400, 'y_stop': 675},
                  {'y_start': 400, 'y_stop': 650},
                  {'y_start': 400, 'y_stop': 625},
                  {'y_start': 400, 'y_stop': 600},
                  {'y_start': 400, 'y_stop': 575},
                  {'y_start': 400, 'y_stop': 562},
                  {'y_start': 400, 'y_stop': 550},
                  {'y_start': 400, 'y_stop': 537},
                  {'y_start': 400, 'y_stop': 525},
                  {'y_start': 400, 'y_stop': 512},
                  {'y_start': 400, 'y_stop': 500},
                  {'y_start': 400, 'y_stop': 487},
                  {'y_start': 400, 'y_stop': 475},
                  {'y_start': 400, 'y_stop': 462},
                  {'y_start': 400, 'y_stop': 450},
                  {'y_start': 400, 'y_stop': 437},
                  {'y_start': 400, 'y_stop': 425},
                  ]

clf = pickle.load(open("linear-SVC-default.p", "rb"))
X_scaler = pickle.load(open("X_scaler.p", "rb"))

box_lists = []
time_average_param = {'averaged_time_steps': 10,
                      'heat_map_threshold': 200}

for f_path, output_path in zip(f_paths, output_paths):
    box_lists = []
    clip = VideoFileClip(f_path)
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_path, audio=False)