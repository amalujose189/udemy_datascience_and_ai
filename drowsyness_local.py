import cv2
import torch
import uuid #unique identifier
import os
import time
from pathlib import Path

# Create output directory if it doesn't exist
output_dir = Path("data/images")
output_dir.mkdir(parents=True, exist_ok=True)
  
from matplotlib import pyplot as plt
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img = "https://ultralytics.com/images/zidane.jpg"
results = model(img)
results.print()

# Render and display using OpenCV (more reliable in .py scripts)
rendered = np.squeeze(results.render())          # RGB image with boxes
rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)  # convert for cv2
cv2.imshow('Static Image Detection', rendered_bgr)
cv2.waitKey(0)          # wait for any key press before continuing
cv2.destroyAllWindows()




cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Camera opened successfully. Press 'q' to exit.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
 #   results.render()  # draws bounding boxes on frame
    output_frame = results.render()[0]
    
    # Convert from RGB to BGR for OpenCV display
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    
    # Save image to data/images folder
    frame_count += 1
    img_path = output_dir / f"detection_{frame_count:05d}.jpg"
    cv2.imwrite(str(img_path), output_frame)
    
    # Show live feed
    cv2.imshow('Drowsiness Detection', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press Q to quit
        break

cap.release()
cv2.destroyAllWindows()
IMAGES_PATH=os.path.join('data', 'images')
labels=['awake', 'drowsy']
number_images=20

cap=cv2.VideoCapture(0)
for label in labels:
    print("collecting images for {}".format(label))
    time.sleep(5)
    for imgnum in range(number_images):
        print("collecting image {}".format(imgnum))
         #webcam feed
        ret, frame=cap.read()
         #naming out image path
        imgname=os.path.join(IMAGES_PATH, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        #writes out theimage path
        cv2.imwrite(imgname, frame) 
        #render to the screen
        cv2.imshow('Image Collection', frame)
        time.sleep(2)
for label in labels:
    print("collecting images for {}".format(label))
    for imgnum in range(number_images):
        print("collecting image {}".format(imgnum))
       
        ret, frame=cap.read()
       
        imgname=os.path.join(IMAGES_PATH, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        print(imgname)