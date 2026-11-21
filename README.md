# Real-Time Object Detection (YOLOv8 + OpenCV)

A real-time object detection app that uses YOLOv8 and OpenCV to detect 80+ object classes from a webcam feed, with live FPS, bounding boxes and confidence scores.

## Features

- Live webcam detection with YOLOv8 (80+ COCO classes: person, phone, car, dog, bottle...)
- Bounding boxes, class labels and confidence scores on screen
- Real-time FPS counter
- Press **S** to save a snapshot of a detection, **Q** to quit

## Tech Stack

- Python
- Ultralytics YOLOv8
- OpenCV

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python detect.py
```

The first run downloads the YOLOv8n model weights automatically.

## How it works

The webcam frame is passed to YOLOv8 each frame. The model returns detected objects with class and confidence; OpenCV draws the boxes and labels and shows the annotated frame live.
