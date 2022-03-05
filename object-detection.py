from typing import Tuple
from dataclasses import dataclass
from time import time
import math
import cv2


MAX_DISTANCE = 30


@dataclass
class Rect:
    id: int
    time_found: float
    center: Tuple[int, int]
    rect: Tuple[int, int, int, int]


class ObjTracker():

    def __init__(self, found_callback):
        self.next_id = 1
        self.rects = {}
        self.notified = set()
        self.found_callback = found_callback

    def next(self, rects, frame):
        found_ids = set()
        for rect in rects:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            found_again = False
            for rect in self.rects.values():
                dist = math.hypot(cx - rect.center[0], cy - rect.center[1])

                if dist < MAX_DISTANCE:
                    rect.center = (cx, cy)
                    rect.rect = (x, y, w, h)
                    found_ids.add(rect.id)
                    found_again = True
                    break

            if not found_again:
                self.rects[self.next_id] = Rect(self.next_id, time(), (cx, cy), (x, y, w, h))
                found_ids.add(self.next_id)
                self.next_id += 1

        now = time()
        for id in found_ids:
            rect = self.rects[id]
            if now - rect.time_found > 0.3 and id not in self.notified:
                self.notified.add(id)
                self.found_callback(rect, frame)

        remove = self.rects.keys() - found_ids
        for id in remove:
            del self.rects[id]

        return self.rects.values()

video = cv2.VideoCapture(0, cv2.CAP_V4L2)
video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
# video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=5000, varThreshold=16, detectShadows=True)

def crop(frame, rect):
    x = max(rect[0], 0)
    y = max(rect[1], 0)
    return frame[y:y+rect[3], x:x+rect[2]]

tracker = ObjTracker(lambda rect, frame: cv2.imshow(str(rect.id), crop(frame, rect.rect)))

while True:
    ret, frame = video.read()
    if ret:
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            PADDING = 40
            x, y, w, h = cv2.boundingRect(contour)
            x -= PADDING
            y -= PADDING
            w += PADDING * 2
            h += PADDING * 2
            area = cv2.contourArea(contour)
            if area > 1000 and area < 10000:
                detections.append([x, y, w, h])
        output = frame if True else cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        for rect in tracker.next(detections, frame):
            [x, y, w, h] = rect.rect
            cv2.rectangle(output, (x, y), (x + w, y + h),
                            (0, 255, 0), 3)
            cv2.putText(output, str(rect.id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('mask', output)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
