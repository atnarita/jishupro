#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

capture = cv2.VideoCapture(1)

cv2.namedWindow("Capture")

while True:
    ret, frame = capture.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray_img, 50, 110)

    cv2.imshow("Capture", frame)
    cv2.imshow("Gray image", gray_img)
    cv2.imshow("Edge image", edge_img)

    c = cv2.waitKey(2)
    if c == 27:
        break

capture.release()
cv2.destroyAllWindows()
