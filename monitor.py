# -*- coding: utf-8 -*-
import sys
import os
import shutil
import json
import urllib.parse
import urllib.request
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

    for i in range(param["capture"]["frames_skip"]):
        cap.read()

    image = np.zeros((param["capture"]["cam_height"],
                      param["capture"]["cam_width"], 3), np.float32)
    cnt = 0
    for i in range(param["capture"]["frames_capture"]):
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
        aruco.drawDetectedMarkers(img_marker, corners, ids, (0, 255, 0))
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

            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, M, (sOut, sOut))

            # 終了判定用マーカー
            if mid[0] == 0:
                # マーカ座標を基準に7セグLED領域を切り出し
                patch_x = sFrm + param['marker_pc']['offset_x']
                patch_y = sFrm + param['marker_pc']['offset_y']
                patch_pc = dst[
                    patch_y: patch_y + param['marker_pc']['height'],
                    patch_x: patch_x + param['marker_pc']['width']
                ]

    return patch_pc


# 画像パッチから電源状態判定(輝度が大きい場合には電源ONと判定)
#   imgはグレースケール(dtype=uint8)
#   返り値：-1=マーカ未発見、0=輝度小(電源OFF)、1=輝度大(電源ON)
def isPowerOn(img):
    if img is None:
        return -1, img, 0.0
    else:
        h, w = img.shape[:2]
        #img_blur = cv2.GaussianBlur(img,(5,5),0)
        ratio = np.sum(img > 128) * 1.0 / img.size
        tmp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(tmp, "%2.2f" % (ratio), (1, h-1),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))

        if ratio > 0.2:
            return 1, tmp, ratio
        else:
            return 0, tmp, ratio


# iftttでイベント送信
def sendIftttEvent(ratioNow, ratioOld):
    value = "{:.2f} -> {:.2f}".format(ratioOld, ratioNow)
    obj = {"value1": value}
    json_data = json.dumps(obj).encode("utf-8")
    #data = urllib.parse.urlencode(obj)
    headers = {"Content-Type" : "application/json"}

    request = urllib.request.Request(param['ifttt']['url'], data=json_data, headers=headers)
    with urllib.request.urlopen(request) as res:
        body = res.read().decode("utf-8")


if __name__ == "__main__":

    # 設定を取得
    if not os.path.exists('init.json'):
        print("error!")
        sys.exit(1)
    param = json.load(open('init.json', 'r'))

    # 処理履歴を取得
    if os.path.exists('log.json'):
        log = json.load(open('log.json', 'r'))
    else:
        log = {}
        now = datetime.now()
        log['datetime'] = now.strftime("%Y/%m/%d %H:%M:%S")
        log['unixtime'] = int(now.timestamp())
        log['ratio'] = 0.0
        log['powerOn'] = -1

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
    patch_pc = cropPatchImages(gray, param)
    if patch_pc is not None:
        cv2.imwrite('log_patch_pc.jpg', patch_pc)

    # 洗濯機の電源状態を判定
    powerOn, patch_pc_b, ratio = isPowerOn(patch_pc)
    if patch_pc_b is not None:
        cv2.imwrite('log_patch_pc_b.jpg', patch_pc_b)

    # 現在時刻を取得
    now = datetime.now()
    now_ut = int(now.timestamp())
    now_dt = now.strftime("%Y/%m/%d %H:%M:%S")
    now_fn = now.strftime("%Y%m%d-%H%M%S")

    # 終了判定結果に基づいて通知
    if powerOn == 0:
        if log['powerOn'] == 1:
            if IS_DEBUG is False:
                sendIftttEvent(ratio, log['ratio'])
        log['powerOn'] = 0
    elif powerOn == 1:
        log['powerOn'] = 1
    log['datetime'] = now_dt
    log['unixtime'] = now_ut
    log['ratio'] = ratio

    # 処理履歴を保存
    f = open('log.json', 'w')
    json.dump(log, f, indent=2)
