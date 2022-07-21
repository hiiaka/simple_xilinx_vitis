import numpy as np
import cv2
import os

def calib_input(iter):
  calib_images = []
  files = os.listdir("./images")
  for file in files[iter * 10: (iter + 1) * 10]:
    image = cv2.imread("./images/" + file, 1)
    height, width = image.shape[:2]
    scale = min(416 / width, 416 / height)
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    new_h, new_w = resized_image.shape[:2]
    dx = (416 - new_w + 1) // 2
    dy = (416 - new_h + 1) // 2
    embed_image = np.ones((416, 416, 3)) * 128
    for h in range(416):
      for w in range(416):
        for c in range(3):
          if (w >= dx) and (w < 416 - dx) \
            and (h >= dy) and (h < 416 - dy):
            embed_image[h, w, 2 - c] \
              = resized_image[h - dy, w - dx, c]
    embed_image /= 255
    calib_images.append(embed_image)
  return {"input_1": calib_images}
