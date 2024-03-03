


# COSC428 Project - Object Detection YOLOv4-tiny and YOLOv8


# Ingredients

1. SVM - Extracting HOG features
2. YOLO MODELS
   A. Build and train a small network using YOLOv8
   B. Object detection using YOLOv4 and YOLOv8



1.  SVM HOG feature testing - SVM FOLDER

   This tests feature extraction with direction using HOG features

   1. download the car images and non car images and save in distinct files or 
   alternatively download from the https://github.com/udacity/CarND-Vehicle-Detection/tree/master/test_images
   2. import the libraries
   3. Update the directory of the "car_images" &  "no_car_images" with the file path in both image.py files to turn images to grey colour and then to extract features

2. YOLO MODELS

   A. Build and train a network using YOLOv8 nano model

   1. upload the vehicle dataset to a annotating tool and manually annotate by selecting the boundary box for objects in each image (public websites can be used to annotate. i.e. CVAT.ai )
   2. download teh annotated files and save under 'labels' 
   3. upload orginal images to validate under images folder
   4. create config.yaml file and update the paths for images to train
   5. import the libraries
   6. now pass the config.yaml file path to build_model.py to train the dataset
   6. run the program and file will be saved in your directory

   B. Vehicle Detection system - Comaprison of YOLOv4-tiny vs YOLOv8

   1. download the coco.names file (https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true')
   2. for yolov4 model, download both cfg and wight files (https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master)
   3. for yolov8 model download the pretrained file, for this training set nano file is downloaded 
   4. in each model give the correct path for both configuration and weights files (#24 & #25 lines in YOLOv4_main.py file and in #8 line in YOLOv8_main.py)
   5. import the libraries 
   6. run the program and observe the bounding boxes, and objecctness score
   7. press "q" to close the camera window in YOLOv4
   8. press "ctrl + c" to close the camera window in YOLOv8

    for vedio I/O error, please adjust the cv.VideoCapture(0) to cv.VideoCapture(-1) for camera selection. 
    to change the accuracy of the object shown change the confThreshold = " " value
    to change the number of frames shown change the nmsThrehold d = " " value 

   