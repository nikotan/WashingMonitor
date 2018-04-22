# -*- coding: utf-8 -*-
import sys, os, shutil
import urllib, urllib2, json
from datetime import datetime
import cv2
import numpy as np

IS_DEBUG = False


# USBカメラからカラー画像を取得
#   返り値：カラー画像(dtype=float32)
def captureImage(param):
  cap = cv2.VideoCapture(param["capture"]["cam_port"])
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  param["capture"]["cam_width"])
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, param["capture"]["cam_height"])
  
  for i in xrange(param["capture"]["frames_skip"]):
    cap.read()
  
  image = np.zeros((param["capture"]["cam_height"], param["capture"]["cam_width"], 3), np.float32)
  cnt = 0
  for i in xrange(param["capture"]["frames_capture"]):
    ret, im = cap.read()
    if ret == True:
      cnt += 1
      image += im
  
  cap.release()

  if cnt > 0:
    image /= cnt

  return image


# 入力画像からマーカを基準に7セグLED領域を見つけて切り出し
#   imgはグレースケール(dtype=uint8)
def cropPatchImages(img, param):
  aruco = cv2.aruco
  dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
  corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
  
  if IS_DEBUG:
    img_marker = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    aruco.drawDetectedMarkers(img_marker, corners, ids, (0,255,0))
    cv2.imwrite('debug_marker.jpg', img_marker)

  patch_pc = None
  patch_min = None
  
  if ids is not None:
    sMkr = param['marker']['size']
    sFrm = sMkr * param['marker']['nFrame']
    sOut = 2 * sFrm + sMkr

    for mid, mcorner in zip(ids, corners):
      pts1 = np.float32(mcorner)
      pts2 = np.float32([
        [sFrm, sFrm],
        [sFrm + sMkr, sFrm],
        [sFrm + sMkr, sFrm + sMkr],
        [sFrm, sFrm + sMkr]])
      
      M = cv2.getPerspectiveTransform(pts1,pts2)
      dst = cv2.warpPerspective(img, M, (sOut, sOut))
      
      #終了判定用マーカー
      if mid[0] == 0:
        #マーカ座標を基準に7セグLED領域を切り出し
        patch_x = sFrm + param['marker_pc']['offset_x']
        patch_y = sFrm + param['marker_pc']['offset_y']
        patch_pc = dst[
          patch_y : patch_y + param['marker_pc']['height'],
          patch_x : patch_x + param['marker_pc']['width']
        ]

      #残り時間判定用マーカー
      elif mid[0] == 1:
        #マーカ座標を基準に7セグLED領域を切り出し
        patch_x = sFrm + param['marker_min']['offset_x']
        patch_y = sFrm + param['marker_min']['offset_y']
        patch_min = dst[
          patch_y : patch_y + param['marker_min']['height'],
          patch_x : patch_x + param['marker_min']['width']
        ]

  return patch_pc, patch_min


# 入力画像から7セグLED領域矩形を取得
#   imgはグレースケール(dtype=uint8)
def getBoundary(img):
  ave = np.average(img)
  img_blur = cv2.GaussianBlur(img,(5,5),0)
  img_blur[img_blur > ave] = ave
  th, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  if IS_DEBUG:
    cv2.imwrite('debug_patch_blur.jpg', img_blur)
    cv2.imwrite('debug_patch_bin.jpg', img_bin)

  img_bin = cv2.bitwise_not(img_bin)
  ret, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  s_max = -1
  x0 = -1
  y0 = -1
  x1 = -1
  y1 = -1
  for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    s = w * h
    if s > s_max:
      s_max = s
      x0 = x
      y0 = y
      x1 = x + w
      y1 = y + h

  return x0, y0, x1, y1, th


# 画像パッチから洗濯終了判定(輝度が大きい場合には終了と判定)
#   imgはグレースケール(dtype=uint8)
#   返り値：-1=マーカ未発見、0=洗濯非終了、1=洗濯終了
def isFinished(img):
  if img is None:
    return -1, img
  else:
    x0, y0, x1, y1, th = getBoundary(img)
    img_blur = cv2.GaussianBlur(img,(5,5),0)
    msk_blur = img_blur[y0 : y1, x0 : x1]
    th_msk, msk_bin = cv2.threshold(msk_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if IS_DEBUG:
      cv2.imwrite('debug_msk_blur.jpg', msk_blur)
      cv2.imwrite('debug_msk_bin.jpg', msk_bin)
      print("th = %s, th_msk = %s" % (th, th_msk))
    
    if th_msk < 50.0:
      th_msk = 50.0
    ratio = np.sum(msk_blur > th_msk) * 1.0 / msk_blur.size
    if IS_DEBUG:
      print("--> th_msk = %s" % (th_msk))
      print("    ratio = %s" % (ratio))

    tmp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(tmp, (x0, y0), (x1-1, y1-1), (0,0,255), 1)
    cv2.putText(tmp, "%2.2f" % (ratio), (x0, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255))

    if ratio > 0.2:
      return 1, tmp
    else:
      return 0, tmp


# iftttでイベント送信
def sendIftttEvent():
  obj = {"value1" : "洗濯が終わっています"}
  json_data = json.dumps(obj).encode("utf-8")
  
  data = urllib.urlencode(obj);
  request = urllib2.Request(param['ifttt']['url'], data)
  response = urllib2.urlopen(request)



if __name__ == "__main__": 

  # 設定を取得
  if not os.path.exists('init.json'):
    print "error!"
    sys.exit(1)
  param = json.load(open('init.json', 'r'))

  # 処理履歴を取得
  if os.path.exists('log.json'):
    log = json.load(open('log.json', 'r'))
  else:
    log = {}
    now = datetime.now()
    log['datetime'] = now.strftime("%Y/%m/%d %H:%M:%S")
    log['unixtime'] = long(datetime.now().strftime('%s'))
    log['count'] = -1

  img = None
  if len(sys.argv) > 1:
    # 引数があった場合(テスト用)
    filename = sys.argv[1]
    path, ext = os.path.splitext(filename)
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
  else:
    # 引数がなかった場合(カメラから画像を取得)
    img = captureImage(param)
  cv2.imwrite('log_image.jpg', img)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = np.uint8(gray)

  # 7セグLED領域矩形を取得
  patch_pc, patch_min = cropPatchImages(gray, param)
  if patch_pc is not None:
    cv2.imwrite('log_patch_pc.jpg', patch_pc)
  if patch_min is not None:
    cv2.imwrite('log_patch_min.jpg', patch_min)

  # 洗濯終了を判定
  finished, patch_pc_b = isFinished(patch_pc)
  if patch_pc_b is not None:
    cv2.imwrite('log_patch_pc_b.jpg', patch_pc_b)
  
  # 現在時刻を取得
  now = datetime.now()
  now_ut = long(now.strftime('%s'))
  now_dt = now.strftime("%Y/%m/%d %H:%M:%S")
  now_fn = now.strftime("%Y%m%d-%H%M%S")

  # 終了判定結果に基づいて通知
  if finished == 1:
    if log['count'] == -1:
      if IS_DEBUG is False:
        sendIftttEvent()
      log['datetime'] = now_dt
      log['unixtime'] = now_ut
      log['count'] = 1
    else:
      dsec = now_ut - log['unixtime']
      dsec_th = log['count'] * param['notify']['interval_sec']
      if log['count'] < param['notify']['max_count'] and dsec > dsec_th:
        sendIftttEvent()
        log['count'] = 1 + dsec / param['notify']['interval_sec']
    shutil.copy('log_image.jpg', 'log/%s.jpg' % (now_fn))
    shutil.copy('log_patch_pc.jpg', 'log/%s_pc.jpg' % (now_fn))
    shutil.copy('log_patch_pc_b.jpg', 'log/%s_pc_b.jpg' % (now_fn))
  elif finished == 0:
    log['count'] = -1

  # 処理履歴を保存
  f = open('log.json', 'w')
  json.dump(log, f, indent=2)
  