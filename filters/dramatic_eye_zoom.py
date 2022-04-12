import cv2
import numpy as np

current_center_x = None
current_center_y = None

prev_box = None

'''
head_follow(image, results[, print]) -> Image

@brief Uses detection data to pan/zoom towards the barycenter of detected faces

@param img input Image
@param results fade detection data from original capture
'''
def dramatic_eye_zoom(image, results):
  global current_center_x, current_center_y, prev_box

  width = image.shape[1]
  height = image.shape[0]

  if prev_box is None:
    prev_box = (0,0,width,height)
  
  
  t = ''
  with open('control.txt') as f:
    t = f.read().strip()
  
  if t and results.detections:
    detection = results.detections[-1]
    box = detection.location_data.relative_bounding_box
    x = box.xmin * width
    y = box.ymin * height
    x2 = x + box.width * width
    y2 = y + (box.height * height)/2
    x -= width * 0.1
    x2 += width * 0.1
  else:
    x = 0
    y = 0
    x2 = width
    y2 = height

  cur_box = (
    max(0,x),
    max(0,y),
    min(x2, image.shape[1]),
    min(y2, image.shape[0])
  )
  
  background = np.zeros(image.shape, dtype=np.uint8)
  
  lerp_box = [prev_box[i] * .90 + cur_box[i] * .1 for i in range(4)]
  cropped = image[int(lerp_box[1]):int(lerp_box[3]), int(lerp_box[0]):int(lerp_box[2])]
  yx_ratio = cropped.shape[0] / cropped.shape[1]
  cropped = cv2.resize(cropped, (background.shape[1], int(background.shape[1] * yx_ratio)))

  center_y = background.shape[0] / 2
  offset_y = cropped.shape[0] / 2
  overlay_y = int(center_y - offset_y)

  background[overlay_y:overlay_y+cropped.shape[0],0:cropped.shape[1]] = cropped[:,:]
  
  prev_box = lerp_box
  return background