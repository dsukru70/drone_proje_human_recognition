import cv2 as cv
import imutils
import numpy as np
import torch
from centroidtracker import CentroidTracker
from itertools import combinations
import math
from djitellopy import Tello

# YOLOv5 modelini yükleyin
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

izci = CentroidTracker(maxDisappeared=40, maxDistance=50)

def main():
    # Tello drone'a bağlan
    tello = Tello()
    tello.connect()

    # Video akışını başlat
    tello.streamon()

    while True:
        # Tello'dan görüntü al
        frame = tello.get_frame_read().frame
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # YOLOv5 ile insan tespiti
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        rects = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0:  # 0, YOLOv5'te 'person' sınıfını temsil eder
                rects.append([int(x1), int(y1), int(x2), int(y2)])

        rects = np.array(rects)
        rects = rects.astype(int)
        rects = non_max_suppression_fast(rects, 0.3)

        mesafe_deposu = dict()
        objects = izci.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)

            mesafe_deposu[objectId] = (cX, cY, x1, y1, x2, y2)

        kirmizi_kare_listesi = []
        for (id1, p1), (id2, p2) in combinations(mesafe_deposu.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            mesafe = math.sqrt(dx * dx + dy * dy)
            if mesafe < 75.0:
                if id1 not in kirmizi_kare_listesi:
                    kirmizi_kare_listesi.append(id1)
                if id2 not in kirmizi_kare_listesi:
                    kirmizi_kare_listesi.append(id2)

        for id, box in mesafe_deposu.items():
            if id in kirmizi_kare_listesi:
                cv.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
            else:
                cv.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)

        cv.imshow("sosyal_mesafe", frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

    # Video akışını durdur
    tello.streamoff()
    cv.destroyAllWindows()

def non_max_suppression_fast(kutular, ust_uste):
    try:
        if len(kutular) == 0:
            return []

        if kutular.dtype.kind == "i":
            boxes = kutular.astype("float")

        pick = []

        x1 = kutular[:, 0]
        y1 = kutular[:, 1]
        x2 = kutular[:, 2]
        y2 = kutular[:, 3]

        kutu = (x2 - x1 + 1) * (y2 - y1 + 1)
        cakisma = np.argsort(y2)

        while len(cakisma) > 0:
            last = len(cakisma) - 1
            i = cakisma[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[cakisma[:last]])
            yy1 = np.maximum(y1[i], y1[cakisma[:last]])
            xx2 = np.minimum(x2[i], x2[cakisma[:last]])
            yy2 = np.minimum(y2[i], y2[cakisma[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            ortusmus = (w * h) / kutu[cakisma[:last]]

            cakisma = np.delete(cakisma, np.concatenate(([last],
                                                         np.where(ortusmus > ust_uste)[0])))

        return kutular[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


main()
