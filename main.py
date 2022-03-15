import os
import cv2
import mediapipe as mp
import numpy as np
import pyfakewebcam
mp_face_detection = mp.solutions.face_detection

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

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.3) as face_detection:

  threshold_counter = 0
  current_center_x = capture_width / 2
  current_center_y = capture_height / 2
  prev_f = 1.4

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


    orig_shape = image.shape
    start_x = image.shape[1]
    end_x = 0
    start_y = image.shape[0]
    end_y = 0

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        box = detection.location_data.relative_bounding_box
        start_x = min(int(box.xmin * image.shape[1]), start_x)
        end_x = max(int((box.xmin + box.width) * image.shape[1]), end_x)
        start_y = min(int(box.ymin * image.shape[0]), start_y)
        end_y = max(int((box.ymin + box.width) * image.shape[0]), end_y)

    background = np.zeros(image.shape, dtype=np.uint8)

    if results.detections:
      center_x = (end_x + start_x) // 2
      center_y = (end_y + start_y) // 2

      # new_f = max(1.4, 1280 / (end_x - start_x) / 4)
      f = 1.4
      # f = new_f * .05 + prev_f *.95
      prev_f = f
    else:
      print("not found, keep current settings")
      center_x = current_center_x
      center_y = current_center_y
      f = prev_f

    if abs(current_center_x - center_x) > 100 or abs(current_center_y - center_y) > 100:
      threshold_counter += 1
    if abs(current_center_x - center_x) < 50 and abs(current_center_y - center_y) < 50:
      threshold_counter = 0

    if threshold_counter < 20:
      center_x = current_center_x
      center_y = current_center_y
      f = prev_f

    current_center_x = int(center_x * .05 + current_center_x *.95)
    current_center_y = int(center_y * .05 + current_center_y *.95)


    delta_x = background.shape[1] // 2 - current_center_x
    delta_y = background.shape[0] // 2 - current_center_y

    delta_x_max = int(image.shape[1] * (1 - 1 / f) / 2)

    if delta_x > 0:
      if delta_x > delta_x_max:
        delta_x = delta_x_max
      background[:,delta_x:] = image[:,0:image.shape[1]-delta_x]
    else:
      if delta_x < -delta_x_max:
        delta_x = -delta_x_max
      background[:,0:image.shape[1]+delta_x] = image[:,-delta_x:]


    delta_y_max = int(image.shape[0] * (1 - 1 / f) / 2)

    if delta_y > 0:
      if delta_y > delta_y_max:
        delta_y = delta_y_max
      background[delta_y:,:] = background[0:image.shape[0]-delta_y,:]
    else:
      if delta_y < -delta_y_max:
        delta_y = -delta_y_max
      background[0:image.shape[0]+delta_y,:] = background[-delta_y:,:]

    background = cv2.resize(background, (int(background.shape[1] * f), int(background.shape[0] * f)))

    s_x = (background.shape[1] // 2 - orig_shape[1] // 2)
    s_y = (background.shape[0] // 2 - orig_shape[0] // 2)
    background = background[s_y:s_y+orig_shape[0],s_x:s_x + orig_shape[1]]

    image = background
    
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
