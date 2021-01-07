#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

capture = cv2.VideoCapture(1)

cv2.namedWindow("Capture")

while True:
    ret, frame = capture.read()

    #cv2.putText(frame, "Detected!", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, 8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray_np = cv2.bitwise_not(gray)
    edge = cv2.Canny(gray, 50,110)
    edge_np = cv2.bitwise_not(edge)
    lines = cv2.HoughLinesP(edge,rho=1,theta=np.pi/180, threshold=20, minLineLength=100,maxLineGap=50)

    # x1, y1, x2, y2 = lines[0,0]
    # red_line_img = cv2.line(edge_np,(x1,y1), (x2,y2), (0,0,255), 3)
    cnt = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1-y2) < 30 and (x1-x2)**2+(y1-y2)**2 > 300**2 and 330<y1 and y1<430:
            red_lines_img = cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(frame, "Take Care!", (100, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 5, 8)
            break
        if cnt ==10:
            break
        cnt += 1
        # 赤線を引く


    cv2.imshow("Capture", frame)

    #print(lines)
    #height, width, channels = frame.shape[:3]
    #print("width: " + str(width))#640
    #print("height: " + str(height))#480

    c = cv2.waitKey(2)
    if c == 27:
        break

capture.release()
cv2.destroyAllWindows()
