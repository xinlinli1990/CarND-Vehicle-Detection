import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

from skimage.feature import hog
from sklearn.utils import shuffle

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs,
                     cspace='RGB', 
                     spatial_size=(32, 32),
                     hist_bins=32, 
                     hist_range=(0, 256), 
                     orient=9,
                     pix_per_cell=8, 
                     cell_per_block=2, 
                     hog_channel=0):

    if spatial_size == 0:
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

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features

        if has_spatial_feature:
            spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() also with a color space option now
        if has_hist_feature:
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Call get_hog_features() with vis=False, feature_vec=True
        if has_hog_features:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                         orient,
                                                         pix_per_cell,
                                                         cell_per_block,
                                                         vis=False,
                                                         feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel],
                                                orient,
                                                pix_per_cell,
                                                cell_per_block,
                                                vis=False,
                                                feature_vec=True)

        image_feature = np.array([]).reshape(0)

        if has_spatial_feature:
            image_feature = np.concatenate((image_feature, spatial_features), axis=0)
        if has_hist_feature:
            image_feature = np.concatenate((image_feature, hist_features), axis=0)
        if has_hog_features:
            image_feature = np.concatenate((image_feature, hog_features), axis=0)
        # Append the new feature vector to the features list
        # features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        features.append(image_feature)
        
    # Return list of feature vectors
    return features


def run(i_cspace=0,
        spatial=32,
        histbin=32,
        orient=9,
        pix_per_cell=8,
        cell_per_block=2,
        hog_channel=3):
    # Read in car and non-car images
    car_image_paths = []
    car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/*/*.png'))
    # car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/GTI_Far/*.png'))
    # car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/GTI_Left/*.png'))
    # car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/GTI_Right/*.png'))
    # car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/GTI_MiddleClose/*.png'))
    # car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/KITTI_extracted/*.png'))

    notcar_image_paths = []
    notcar_image_paths.extend(glob.glob('../dataset/non-vehicles/non-vehicles/*/*.png'))
    # notcar_image_paths.extend(glob.glob('../dataset/non-vehicles/non-vehicles/GTI/*.png'))
    # notcar_image_paths.extend(glob.glob('../dataset/non-vehicles/non-vehicles/Extras/*.png'))

    car_image_paths = shuffle(car_image_paths)
    notcar_image_paths = shuffle(notcar_image_paths)

    cars = car_image_paths#[:2000]
    notcars = notcar_image_paths#[:2000]

    # play with these values to see how your classifier
    # performs under different binning scenarios
    cspaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV']
    # i_cspace = 3
    # spatial = 32
    # histbin = 32
    # orient = 9
    # pix_per_cell = 8
    # cell_per_block = 2
    # hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    if hog_channel == 3:
        hog_channel = 'ALL'

    #t_start=time.time()
    car_features = extract_features(cars, cspace=cspaces[i_cspace], spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256),orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel)

    #t_extract_car=time.time()
    #print(round(t_extract_car-t_start, 5), 'Seconds to extract car features')
    notcar_features = extract_features(notcars, cspace=cspaces[i_cspace], spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256), orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel)
    #t_extract_not_car=time.time()
    #print(round(t_extract_not_car-t_extract_car, 5), 'Seconds to extract non-car features')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # print('Using spatial binning of:',spatial,
    #     'and', histbin,'histogram bins')
    # print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    # svc = LinearSVC()
    # clf = LinearSVC()
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(X_train, y_train)
    # Check the training time for the SVC
    # t=time.time()

    #svc.fit(X_train, y_train)
    # t2 = time.time()
    # print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    # return round(svc.score(X_test, y_test), 4)
    return round(clf.score(X_test, y_test), 4)
    # print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    # t=time.time()
    # n_predict = X_test.shape[0]
    # print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    # print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    # t2 = time.time()
    # print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')