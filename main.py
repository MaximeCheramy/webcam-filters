import os
import cv2
import mediapipe as mp
import numpy as np
import pyfakewebcam

from filters.head_follow import head_follow

SOURCE_CAMERA = int(os.environ.get('SOURCE_CAMERA', '0'))
VIRTUAL_CAMERA = int(os.environ.get('VIRTUAL_CAMERA', '8'))
VIRTUAL_WIDTH = int(os.environ.get('VIRTUAL_WIDTH', '1280'))
VIRTUAL_HEIGHT = int(os.environ.get('VIRTUAL_HEIGHT', '720'))

camera = pyfakewebcam.FakeWebcam(f"/dev/video{VIRTUAL_CAMERA}", VIRTUAL_WIDTH, VIRTUAL_HEIGHT)

# For webcam input:
cap = cv2.VideoCapture(SOURCE_CAMERA)
cap.set(3, VIRTUAL_WIDTH)
cap.set(4, VIRTUAL_HEIGHT)
cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

capture_width = cap.get(3)
capture_height = cap.get(4)

if capture_width != VIRTUAL_WIDTH or capture_height != VIRTUAL_HEIGHT:
  print("Warning: capture device does not support requested frame size")

win_name = 'MediaPipe Face Detection'
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.3) as face_detection:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    image = head_follow(image, results)
    
    windowVisible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow(win_name, cv2.flip(image, 1))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[0] != VIRTUAL_WIDTH or image.shape[1] != VIRTUAL_HEIGHT:
      image = cv2.resize(image, (VIRTUAL_WIDTH, VIRTUAL_HEIGHT))
    camera.schedule_frame(image)
    if not windowVisible or cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
