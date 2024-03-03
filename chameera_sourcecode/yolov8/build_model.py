import cv2 as cv
from ultralytics import YOLO
 
 #https://github.com/ultralytics/ultralytics

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch using the nano version
# Use the model
 

model.train(data="/Users/user1/Downloads/CV/Chameera/yolov8/config.yaml", epochs=4)  # train the model 