# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np

if len(sys.argv) > 1:
  filename = sys.argv[1]

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

size = 20
margin = 2
m0_offset_x = 4
m0_offset_y = 5
m0_width = 25
m0_height = 18
m1_offset_x = -29
m1_offset_y = 20
m1_width = 40
m1_height = 22

path, ext = os.path.splitext(filename)

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
cv2.imwrite(path + '_out' + ext, img)

finished = False

for mid, mcorner in zip(ids, corners):
  pts1 = np.float32(mcorner)
  pts2 = np.float32([
    [size*margin,size*margin],
    [size*(margin+1),size*margin],
    [size*(margin+1),size*(margin+1)],
    [size*margin,size*(margin+1)]])

  M = cv2.getPerspectiveTransform(pts1,pts2)
  dst = cv2.warpPerspective(img, M, (size*(margin*2+1),size*(margin*2+1)))

  if mid[0] == 0:
    msk = dst[
      size*(margin+1)+m0_offset_y : size*(margin+1)+m0_offset_y+m0_height,
      size*(margin+1)+m0_offset_x : size*(margin+1)+m0_offset_x+m0_width
    ]
    cv2.imwrite('%s_%s_result%s' % (path, mid[0], ext), msk)
    if np.sum(msk > 128) * 1.0 / msk.size > 0.2:
      finished = True

  elif mid[0] == 1:
    cv2.imwrite(
      '%s_%s_result%s' % (path, mid[0], ext),
      dst[
        size*(margin+1)+m1_offset_y : size*(margin+1)+m1_offset_y+m1_height,
        size*(margin+1)+m1_offset_x : size*(margin+1)+m1_offset_x+m1_width
      ]
    )

print finished

