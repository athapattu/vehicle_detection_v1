import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog

# import training dataset of cars and non-cars
import glob

# sources  https://github.com/udacity/CarND-Vehicle-Detection/tree/master/test_images
car_images = glob.glob("/Users/user1/Downloads/CV/Chameera/svm/train_car/*.jpg")  # import images of cars image files
no_car_images = glob.glob("/Users/user1/Downloads/CV/Chameera/svm/train_nocar/*.jpg")

print(len(car_images))  # print length of all image files
print(len(no_car_images))

# Choose an index to display a specific image
index = 2

# Display the color image
image_color = cv2.imread(car_images[index])
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.title('Color Image')
plt.show()

# Convert the color image to grayscale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Get HOG Features
# applying features to the gray image
features, hog_image = hog(image_gray,
                          orientations = 11, # angles 
                          pixels_per_cell = (16,16), #window
                          cells_per_block = (2,2), # amount of cells
                          transform_sqrt = False, 
                          visualize = True,
                          feature_vector = True) 
# plot
print(features.shape) 
print(hog_image.shape)
plt.imshow(hog_image, cmap= 'gray')
plt.show() 

 