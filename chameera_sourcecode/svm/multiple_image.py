import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
import glob

# import training dataset of cars and non-cars
car_images = glob.glob("/Users/user1/Downloads/CV/Chameera/svm/train_car/*.jpg")  # import images of cars image files
no_car_images = glob.glob("/Users/user1/Downloads/CV/Chameera/svm/train_nocar/*.jpg")

print(len(car_images))  # print length of all image files
print(len(no_car_images)) 

# Hog feature extraction and training dataset
print("Hog feature extraction and training dataset")

# for images with cars

car_hog_accum = []

for car in car_images:
    image_color = mpimg.imread(car)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)  # obtain gray colour

    car_hog_feature, car_hog_img = hog(image_color[:, :, 0],  # apply hog feature
                                      orientations=11,  # angles 
                                      pixels_per_cell=(16, 16),  # window
                                      cells_per_block=(2, 2),  # amount of cells
                                      transform_sqrt=False, 
                                      visualize=True,
                                      feature_vector=True) 
    
    car_hog_accum.append(car_hog_feature)  # append extracted hog feature to the list

# Find the maximum number of features in all images
max_features = max(len(features) for features in car_hog_accum)

# Set the number of features per image to the maximum
for i, features in enumerate(car_hog_accum):
    car_hog_accum[i] = np.pad(features, (0, max_features - len(features)))

# To train ML model need features and labels, output 1 = car detected, all images of no cars then indicate 0 
x_car = np.vstack(car_hog_accum).astype(np.float64)  # numpy vertical stack to stack all features in array
y_car = np.ones(len(x_car))  # creating one length of car

print(x_car.shape) # number of cars, hog features per each image
print(y_car.shape)
print(y_car) # output label will be 1


# for images with no cars

no_car_hog_accum = []

for no_car in no_car_images:
    image_color = mpimg.imread(no_car)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)  # obtain gray colour

    no_car_hog_feature, no_car_hog_img = hog(image_color[:, :, 0],  # apply hog feature
                                      orientations=11,  # angles 
                                      pixels_per_cell=(16, 16),  # window
                                      cells_per_block=(2, 2),  # amount of cells
                                      transform_sqrt=False, 
                                      visualize=True,
                                      feature_vector=True) 
    
    no_car_hog_accum.append(no_car_hog_feature)  # append extracted hog feature to the list

# Find the maximum number of features in all images
max_features = max(len(features) for features in no_car_hog_accum)

# Set the number of features per image to the maximum
for i, features in enumerate(no_car_hog_accum):
    no_car_hog_accum[i] = np.pad(features, (0, max_features - len(features)))

# To train ML model need features and labels, output 1 = car detected, all images of no cars then indicate 0 
x_no_car = np.vstack(no_car_hog_accum).astype(np.float64)  # numpy vertical stack to stack all features in array
y_no_car = np.zeros(len(x_no_car))  # creating one length of car

print(x_no_car.shape) # number of no_cars, hog features per each image
print(y_no_car.shape)
print(y_no_car) # output label will be 0



