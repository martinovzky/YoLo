import cv2
import torch
from ultralytics import YOLO

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' #inference will fall back to CPU for certain operations if MPS is not available

#select device
device = 'mps' if torch.backends.mps.is_available() else 'cpu' #should work on MacOS with M1/M2 chips
print(f"Using device: {device}, with CPU fallback enabled")

#load the office items fine-tuned model 
model = YOLO('runs/train/OfficeItems_yolov8m/weights/best.pt') #best.pt contains the model weights at the best validation performance

#opent the webcam

cap = cv2.VideoCapture(0) #0 is the default camera, change to 1 or 2 if you have multiple cameras

if not cap.isOpened():
    raise Exception("Could not open video device")

#inference loop
while True:
    ret,frame = cap.read() #read a frame from the webcam, ret is True if the frame is read successfully
    if not ret:
        break #if no frame is read, break the loop


    
#run detection at confidence threshold of 0.28
results = model(frame, conf=0.28, device=device)[0] #run the model on the frame, results is a list of detections, conidence treashold is the minnimum confidence score to consider a detection valid
#need to add [0], as the model processes one image at a time and returns it in a list

#draw the results on the frame, i.e. bounding boxes and labels
for *box, conf, cls in results.boxes.data.tolist(): #for each detection, box is a list of [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box) #convert the box coordinates to integers (as they are floats, and we need integers for pixel coordinates)
    label = model.names[int(cls)] #get the label name from the class index
    text = f"{label} {conf:.2f}" #create the text to display on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) #draw a rectangle around the detected object
    cv2.putText(frame,text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #draw the label and confidence score on the frame

#display the frame with the detections
cv2.imshow('Office Items Detection', frame) #show the frame with the detections
if cv2.waitKey(1) == 27: #if the user presses the ESC key, break the loop
    break

#release the webcam and close all windows
cap.release() #release the webcam
cv2.destroyAllWindows() #close all windows


