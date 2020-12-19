#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

capture = cv2.VideoCapture(1)

cv2.namedWindow("Capture")

while True:
    ret, frame = capture.read()

    cv2.putText(frame, "Detected!", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, 8)
    cv2.imshow("Capture", frame)

    c = cv2.waitKey(2)
    if c == 27:
        break

capture.release()
cv2.destroyAllWindows()
