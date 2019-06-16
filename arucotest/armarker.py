# -*- coding: utf-8 -*-

import sys
import cv2

mid = 0
if len(sys.argv) > 1:
  mid = int(sys.argv[1])

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

fileName = "ar%s.png" % mid
generator = aruco.drawMarker(dictionary, mid, 100)
cv2.imwrite(fileName, generator)
