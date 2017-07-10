import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from mpl_toolkits.mplot3d import Axes3D

from sklearn.utils import shuffle

# from skimage.feature import hog
# from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

def plot3d(c0, c1, c2,
           colors_rgb,
           axis_labels=list("RGB"),
           axis_limits=[(0, 255), (0, 255), (0, 255)]):
    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        c0,
        c1,
        c2,
        c=colors_rgb, edgecolors='none')

    return ax  # return Axes3D object for further manipulation

car_image_paths = []
car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/GTI_Far/*.png'))
car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/GTI_Left/*.png'))
car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/GTI_Right/*.png'))
car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/GTI_MiddleClose/*.png'))
car_image_paths.extend(glob.glob('../dataset/vehicles/vehicles/KITTI_extracted/*.png'))

notcar_image_paths = []
notcar_image_paths.extend(glob.glob('../dataset/non-vehicles/non-vehicles/GTI/*.png'))
notcar_image_paths.extend(glob.glob('../dataset/non-vehicles/non-vehicles/Extras/*.png'))

def plot_image_hist(image_paths, ax0, ax1, ax2):

    image_paths = shuffle(image_paths)
    image_paths = image_paths[:1000]

    C0 = np.array([])
    C1 = np.array([])
    C2 = np.array([])
    #BGR = None

    for image_path in image_paths:
        image = cv2.imread(image_path)
        C012 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # if BGR is None:
        #     BGR = (image / 255.).reshape((-1, 3))
        # else:
        #     BGR = np.concatenate((BGR, (image / 255.).reshape((-1, 3))), axis=0)

        C012[:,:,0] = clahe.apply(C012[:,:,0])
        C012[:,:,1] = clahe.apply(C012[:,:,1])
        C012[:,:,2] = clahe.apply(C012[:,:,2])

        C0 = np.concatenate((C0, C012[:, :, 0].ravel()), axis=0)
        C1 = np.concatenate((C1, C012[:, :, 1].ravel()), axis=0)
        C2 = np.concatenate((C2, C012[:, :, 2].ravel()), axis=0)
        
    #H,S,V,BGR = shuffle(H,S,V,BGR)

    ax0.hist(C0, 256, range=(0,255), normed=1)
    ax1.hist(C1, 256, range=(0,255), normed=1)
    ax2.hist(C2, 256, range=(0,255), normed=1)

    return ax0, ax1, ax2
    #return plot3d(H[:4096], S[:4096], V[:4096], BGR[:4096, :], axis_labels=list("HSV"))

#plot_image_3d(car_image_paths)

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 5))

ylim_max = 0.03

ax1, ax2, ax3 = plot_image_hist(car_image_paths, ax1, ax2, ax3)
ax1.set_title('Car Images (H Channel)', fontsize=8)
ax2.set_title('Car Images (S Channel)', fontsize=8)
ax3.set_title('Car Images (V Channel)', fontsize=8)
ax1.set_ylim([0, ylim_max])
ax2.set_ylim([0, ylim_max])
ax3.set_ylim([0, ylim_max])

ax4, ax5, ax6 = plot_image_hist(notcar_image_paths, ax4, ax5, ax6)
ax4.set_title('Non-Car Images (H Channel)', fontsize=8)
ax5.set_title('Non-Car Images (S Channel)', fontsize=8)
ax6.set_title('Non-Car Images (V Channel)', fontsize=8)
ax4.set_ylim([0, ylim_max])
ax5.set_ylim([0, ylim_max])
ax6.set_ylim([0, ylim_max])

plt.savefig('./HSV_CLI.jpg')
plt.show()
