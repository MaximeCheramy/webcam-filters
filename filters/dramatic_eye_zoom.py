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

  orig_shape = image.shape
  width = image.shape[1]
  height = image.shape[0]

  if prev_box is None:
    prev_box = (0,0,width,height)
  
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.detections:
    detection = results.detections[-1]
    box = detection.location_data.relative_bounding_box
    x = box.xmin * width
    y = box.ymin * height
    x2 = x + box.width * width
    y2 = y + (box.height * height)/2

    x -= width * 0.1
    x2 += width * 0.1

    cur_box = (x,y,x2,y2)
    lerp_box = [prev_box[i] * .90 + cur_box[i] * .1 for i in range(4)]
    image = image[int(lerp_box[1]):int(lerp_box[3]), int(lerp_box[0]):int(lerp_box[2])]
    prev_box = lerp_box


  return image