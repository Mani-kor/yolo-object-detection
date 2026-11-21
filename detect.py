# detect.py - Object detection with YOLO + OpenCV
#
# Three ways to use it:
#   python detect.py              -> live webcam detection (Q quits, S saves a snapshot)
#   python detect.py myvideo.mp4  -> detection on a video file
#   python detect.py photo.jpg    -> detection on a single image: draws the boxes,
#                                    shows them in a window, prints what it found,
#                                    and saves the result as result_photo.jpg

import sys    # reads the optional command-line argument (file name)
import time   # for the FPS calculation in video mode
import os     # for building the result_<name>.jpg file name
import cv2    # OpenCV: webcam access, drawing, and windows
from ultralytics import YOLO  # the YOLO model wrapper

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---------- 1. Load the model ----------
# yolov8n.pt is the smallest, fastest YOLOv8 model ("n" = nano), trained on
# the COCO dataset, so it already knows 80 everyday object types (person,
# phone, cup, dog, ...). The weights file downloads automatically on the
# first run. You can later try yolo11n.pt, the newer nano model.
model = YOLO("yolov8n.pt")

# ---------- 2. Shared helper: draw boxes + labels on one frame ----------
def draw_boxes(frame, results):
    """Draw a box and label for every detected object, and print each one."""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # box corners, in pixels
        confidence = float(box.conf[0])         # how sure the model is (0 to 1)
        class_id = int(box.cls[0])              # the object type, as a number
        name = model.names[class_id]            # turn the number into "person", "cup", ...

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{name} {confidence:.0%}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        print(f"  found: {name} ({confidence:.0%})")

# ---------- 3. Image mode: one picture in, one annotated picture out ----------
def run_on_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read the image: {path}")
        sys.exit(1)

    results = model(frame, verbose=False)[0]
    print(f"Detections in {path}:")
    if len(results.boxes) == 0:
        print("  (no objects found)")
    draw_boxes(frame, results)

    # Always save the annotated image: even if the preview window causes
    # trouble on some laptops, the saved file is the reliable result.
    name, _ = os.path.splitext(os.path.basename(path))
    out_name = f"result_{name}.jpg"
    cv2.imwrite(out_name, frame)
    print(f"Saved the annotated image as {out_name}")

    cv2.imshow("Detection result (press any key to close)", frame)
    cv2.waitKey(0)  # wait until any key is pressed
    cv2.destroyAllWindows()

# ---------- 4. Video mode: webcam or a video file, frame by frame ----------
def run_on_video(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Could not open the video source. If you are using the webcam,")
        print("check that no other app is using it and that Windows allows")
        print("camera access (Settings > Privacy > Camera).")
        sys.exit(1)

    prev_time = time.time()  # starting point for the FPS calculation

    while True:
        ok, frame = cap.read()
        if not ok:  # webcam read failed or the video file ended
            break

        # Run YOLO on this frame. verbose=False keeps the console quiet.
        results = model(frame, verbose=False)[0]
        draw_boxes(frame, results)

        # FPS = frames per second: how fast the model runs on your laptop
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        cv2.imshow("Object detection (Q to quit, S to save)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite("snapshot.jpg", frame)
            print("Saved snapshot.jpg")

    cap.release()
    cv2.destroyAllWindows()

# ---------- 5. Choose the mode from the command line ----------
if len(sys.argv) > 1 and sys.argv[1].lower().endswith(IMAGE_EXTENSIONS):
    run_on_image(sys.argv[1])          # an image file was passed
else:
    source = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 = webcam
    run_on_video(source)
