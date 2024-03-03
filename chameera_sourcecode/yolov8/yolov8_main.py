import cv2 as cv
import urllib.request
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

# Use '0' as the source for the default camera
results_webcam = model.predict(source="0", show=True)

confThreshold = 0.5   # to change the accuracy of the object shown change the confThreshold = " " value
nmsThreshold = 0.3 #    to change the number of frames shown change the nmsThrehold d = " " value 


classesFile = 'coco.names' # pass the coco names file in the same directory tto classify the class names
classNames = [] # empty list to store class names
with open(classesFile, 'rt') as f: # opens the class file in read text mode as 'f'
    classNames = f.read().rstrip('\n').split('\n') # use to split class names

print(f"Configuration File: yolov8n.pt") # pass the pretrained nanao file

cap = cv.VideoCapture(0) # grab the avaialbe camera
frame_number = 0  # Initialize frame number

while True:
    success, img = cap.read() # read the frame of the video
    frame_number += 1  # Increment frame number

    # receiving predictions from Ultralytics YOLO model for webcam frames
    results_webcam = model(img)

    # Check if results_webcam list is not empty
    if results_webcam:
        for i, result_dict in enumerate(results_webcam):# Loop through each element in the results_webcam
            # Access the 'boxes' attribute for each dictionary
            img_results = result_dict.boxes.xyxy[0]  # Change 'xyxy' to 'boxes'

            # Check if img_results is not empty
            if img_results.size(0) > 0:
                # Loop through results for each class
                for result in img_results:
                    # Check if confidence threshold is met
                    if result.numel() >= 6:  # Ensure there are at least 6 elements in the result tensor
                        confidence = float(result[4])  # Confidence score is at index 4
                        if confidence > confThreshold:
                            classId = int(result[5])  # Class index is at index 5
                            x, y, w, h = map(int, result[:4])  # Extract bounding box coordinates

                            # Display additional information on the image
                            additional_info = f'Class: {classNames[classId].upper()}, Coordinates: ({x}, {y}), Width: {w}, Height: {h}' # box cordinates
                            cv.putText(img, additional_info, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # font type

                            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2) # detects the top-left-croner and bottom right corner, change colour and thickness
                            cv.putText(img, f'{classNames[classId].upper()} {int(confidence * 100)}%', (x, y - 10), # placing the text on top of the boundary box
                                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2) # color and thickness
  
      # Display the frame number and accuracy on the image
        cv.putText(img, f'Frame: {frame_number}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Image', img) # display an image in window
  
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release() # allows to close the camera capture
    cv.destroyAllWindows() # close the OpenCV windows
 


