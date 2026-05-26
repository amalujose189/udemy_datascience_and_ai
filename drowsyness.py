import matplotlib
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2  #open cv

#load model
model=torch.hub.load('ultralytics/yolov5','yolov5s')
print(model)

# Open camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Camera opened successfully. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame from camera")
        break
    
    # Run YOLOv5 detection on frame
    results = model(frame)
    
    # Render results on frame
    output_frame = results.render()[0]
    
    # Display the frame with detections
    cv2.imshow('Drowsiness Detection', output_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed successfully")
