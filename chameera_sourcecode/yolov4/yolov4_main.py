import cv2 as cv
import numpy as np
import urllib.request
import time

cap = cv.VideoCapture(0)
# width & height of the image
whT = 320

# Download coco.names file
url = 'https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true'
urllib.request.urlretrieve(url, 'coco.names')

confThreshold = 0.5
# (lower value less box)
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Example for the same directory
modelConfiguration = '/Users/user1/Downloads/Chameera_code/chameera_sourcecode/yolov4/yolov4-tiny.cfg'
modelWeights = '/Users/user1/Downloads//Chameera_code/chameera_sourcecode/yolov4/yolov4-tiny.weights'
# ...

# Print paths for debugging
print(f"Configuration File: {modelConfiguration}")
print(f"Weights File: {modelWeights}")

# networkq
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# opencv setup
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# cpu set up
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Variables for calculating frame rate
start_time = time.time()
frame_count = 0

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []  # xywh
    classIds = []  # classID
    confs = []  # confidence value

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    if bbox and classIds and confs:
        indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        indices = indices.flatten()

        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                       (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    success, img = cap.read()

    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()

    # outputNames = [(layersNames[i[0]-1]) for i in net.getUnconnectedOutLayers()]X_X

    # Get the indices of the output layers
    outputNames = net.getUnconnectedOutLayersNames()

    # three different output Rows300 Clo85; R1200;R4800 from 80 class
    # 300 bonding boxes; 85 4 value center x center y w h 1 value confidents object present in the bonding box
    # 80 is the probability of each class in coco
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv.imshow('Image', img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
