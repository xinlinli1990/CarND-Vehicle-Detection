import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

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

hog_features_param = {'orient': 9,
                      'pix_per_cell': 8,
                      'cell_per_block': 2}

#img = cv2.imread("../images/image0749.png")
img = cv2.imread("../images/image400.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

for channel in range(0, 3):
    feature, hog_img = get_hog_features(img[:,:,channel], 9, 8, 2, vis=True, feature_vec=True)
    hog_img = cv2.resize(hog_img, (150,150))
    plt.imsave('../images/notcar_img_' + str(channel) + '.png', cv2.resize(img[:,:,channel], (150,150)), cmap='gray')
    plt.imsave('../images/notcar_hog_' + str(channel) + '.png', hog_img, cmap='gray')
