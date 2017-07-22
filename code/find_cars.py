import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2


import numpy as np
import cv2
from skimage.feature import hog

from scipy.ndimage.measurements import label

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2LAB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, search_windows, hog_features_param, clf, X_scaler):
    box_list = []

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255.0

    # hog feature
    orient = hog_features_param['orient']
    pix_per_cell = hog_features_param['pix_per_cell']
    cell_per_block = hog_features_param['cell_per_block']

    for search_window in search_windows:

        y_start = search_window['y_start']
        y_stop = search_window['y_stop']

        img_tosearch = img[y_start:y_stop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

        imshape = ctrans_tosearch.shape

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        scan_window_size = 64

        x_y_ratio = np.float(imshape[1]) / np.float(imshape[0])
        scale = np.float(imshape[0]) / np.float(scan_window_size)
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(scan_window_size * x_y_ratio), np.int(scan_window_size)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

        nblocks_per_window = (scan_window_size // pix_per_cell) - cell_per_block + 1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps + 1):
            for yb in range(nysteps + 1):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Scale features and make a prediction
                test_features = X_scaler.transform(image_feature.reshape(1, -1))
                test_prediction = clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(scan_window_size * scale)

                    box_list.append(((xbox_left, ytop_draw + y_start), (xbox_left + win_draw, ytop_draw + win_draw + y_start)))
                    # cv2.rectangle(draw_img, (xbox_left, ytop_draw + y_start),
                    #               (xbox_left + win_draw, ytop_draw + win_draw + y_start), (0, 0, 255), 6)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, box_list)
    heat = apply_threshold(heat, 4)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(draw_img), labels)

    return draw_img

# Update car box buffer for video pipeline
def update_car_box_lists(img, search_windows, color_space, spatial_features_param, hist_features_param, hog_features_param,
                         clf, X_scaler, box_lists, average_time_steps):

    draw_img = np.copy(img)

    while len(box_lists) >= average_time_steps:
        box_lists.pop(0)

    box_list = []
    img = img.astype(np.float32) / 255.0

    # hog feature
    orient = hog_features_param['orient']
    pix_per_cell = hog_features_param['pix_per_cell']
    cell_per_block = hog_features_param['cell_per_block']

    # color hist feature
    hist_bins = hist_features_param['hist_bins']

    # color bin feature
    spatial_size = spatial_features_param['spatial_size']

    if spatial_size == (0, 0):
        has_spatial_feature = False
    else:
        has_spatial_feature = True

    if hist_bins == 0:
        has_hist_feature = False
    else:
        has_hist_feature = True

    if orient == 0 or pix_per_cell == 0 or cell_per_block == 0:
        has_hog_features = False
    else:
        has_hog_features = True

    for search_window in search_windows:

        y_start = search_window['y_start']
        y_stop = search_window['y_stop']

        img_tosearch = img[y_start:y_stop, :, :]
        #img_tosearch = cv2.resize(img_tosearch, (np.int(img_tosearch.shape[1] * 1.2), np.int(img_tosearch.shape[0] * 1.2)))

        ctrans_tosearch = convert_color(img_tosearch, conv=color_space) # 1 HSV

        imshape = ctrans_tosearch.shape

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        scan_window_size = 64
        scan_window_scale = 1.2

        x_y_ratio = np.float(imshape[1]) / np.float(imshape[0])
        scale = np.float(imshape[0]) / np.float(scan_window_size) / scan_window_scale
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(scan_window_size * x_y_ratio * scan_window_scale), np.int(scan_window_size * scan_window_scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

        nblocks_per_window = (scan_window_size // pix_per_cell) - cell_per_block + 1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps + 1):
            for yb in range(nysteps + 1):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                if has_hog_features:
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + scan_window_size, xleft:xleft + scan_window_size], (64, 64))

                # x_ori = xleft / (scan_window_size * x_y_ratio * scan_window_scale) * imshape[1]
                # y_ori = ytop / (scan_window_size * scan_window_scale) * imshape[0] + y_start
                # cv2.rectangle(img, (x_ori, y_ori),
                #               (xbox_left + win_draw, ytop_draw + win_draw + y_start), (0, 0, 255), 6)

                # Get color features
                if has_spatial_feature:
                    spatial_features = bin_spatial(subimg, size=spatial_size)

                # Hist features
                if has_hist_feature:
                    hist_features = color_hist(subimg, nbins=hist_bins, bins_range=(0, 256))

                image_feature = np.array([])
                if has_spatial_feature:
                    image_feature = np.concatenate((image_feature, spatial_features), axis=0)
                if has_hist_feature:
                    image_feature = np.concatenate((image_feature, hist_features), axis=0)
                if has_hog_features:
                    image_feature = np.concatenate((image_feature, hog_features), axis=0)

                # Scale features and make a prediction
                test_features = X_scaler.transform(image_feature.reshape(1, -1))
                test_prediction = clf.predict(test_features)

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(scan_window_size * scale)

                if xb is 0 and yb is 0:
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + y_start),
                                  (xbox_left + win_draw, ytop_draw + win_draw + y_start), (0, 0, 255), 5)
                else:
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + y_start),
                                  (xbox_left + win_draw, ytop_draw + win_draw + y_start), (0, 0, 255), 1)
                    pass

                if test_prediction == 1:
                    box_list.append(
                        ((xbox_left, ytop_draw + y_start), (xbox_left + win_draw, ytop_draw + win_draw + y_start)))

    box_lists.append(box_list)
    return box_lists, draw_img

# hog_features_param = {'orient': 9, 'pix_per_cell': 8, 'cell_per_block': 2}
#
# search_windows = [{'y_start': 400, 'y_stop': 575},
#                   {'y_start': 400, 'y_stop': 562},
#                   {'y_start': 400, 'y_stop': 550},
#                   {'y_start': 400, 'y_stop': 537},
#                   {'y_start': 400, 'y_stop': 525},
#                   {'y_start': 400, 'y_stop': 512},
#                   {'y_start': 400, 'y_stop': 500},
#                   {'y_start': 400, 'y_stop': 487},
#                   {'y_start': 400, 'y_stop': 475},
#                   {'y_start': 400, 'y_stop': 462},
#                   {'y_start': 400, 'y_stop': 450},
#                   {'y_start': 400, 'y_stop': 437},
#                   {'y_start': 400, 'y_stop': 425}]
#
# clf = pickle.load(open("linear-SVC-default.p", "rb"))
# X_scaler = pickle.load(open("X_scaler.p", "rb"))
#
# img = mpimg.imread('../test_images/test1.jpg')
#
# result = find_cars(img, search_windows, hog_features_param, clf, X_scaler)
# plt.imshow(result)
# plt.show()