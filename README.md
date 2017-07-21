## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


# **Vehicle Tracking Project**

## Introduction

This project aims to implement a vehicle detection and tracking pipeline based on 
traditional machine learning method. A linear support vector machine (SVM) was trained
with [GTI vehicle image database](https://www.gti.ssr.upm.es/data/Vehicle_database.html)
and vehicle images extracted from [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/).

-- Extract features from dataset	
-- Train classifier with extracted features	
-- Detect vehicle with classifier	

Can be improved if using deep learning methods like Fast RCNN or YOLO.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Video



## Feature extraction

### Histogram of Oriented Gradients (HOG) feature

In computer vision, HOG feature was a state-of-the-art and very popular feature descriptor 
before deep neural network. In CVPR 2005, Dalal et al. presented their pedestrian detection 
pipeline combining HOG feature and linear SVM in [this paper](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf).

The HOG feature can be extracted by this code.
```python
from skimage.feature import hog

hog_features_param = {'orient': 9,
                      'pix_per_cell': 8,
                      'cell_per_block': 2}

# Define a function to return HOG features and visualization
def get_hog_features(img, 
                     orient, 
					 pix_per_cell, 
					 cell_per_block,
					 vis=False, 
					 feature_vec=True):
					 
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
```

For simplicity reason, I used the HOG parameters proposed from the original pedestrian detection paper 
and the results proved that it also works well for vehicle detection. Furthermore, as purposed in the original paper,
 missing rate of detection could be improved by performing the parameter tuning (cell size and block size) 

![](./images/HOG_param.JPG)

After explored different color spaces and hyperparameters, I decided to use YCrCb color space. 

Here is an example of the hog feature extracted in YCrCb color space from one car image and one non-car image . 	
![car-hog](./images/car-hog.jpg)	

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

## Classifier traning

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

## Vehicle detection and tracking with sliding window search

### Sliding Window Search

sliding window image



#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

## Next steps

Try XGBoost, or deep learning approaches like Fast RCNN and YOLO.