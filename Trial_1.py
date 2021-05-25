import cv2
import numpy as np
import time
import winsound
import pandas
from datetime import datetime


video = cv2.VideoCapture(0)
video.set(3, 640)  # Width # reducing resolution for quick processing
video.set(4, 480)  #Height

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)); #Remove noise in MOG2

#creating object
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40);

first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

#a = 1

while True:
    #Capture the video frame
    #a = a + 1
    check, frame = video.read()
    height, width, _ =frame.shape
    #print(height, width)

    #print(check)
    print(frame)

    #roi = frame[340:620, 300:460]

    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    #edged = cv2.Canny(gray, 25, 75)
    fgmask = fgbg.apply(gray);
    #delta_frame = cv2.absdiff(first_frame, gray)
    #thresh_delta = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]
    #thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)

    # apply transformation to remove noise
    fgmask0 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel);

    (cnts, __) = cv2.findContours(fgmask0.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #thresh_delta.copy()

    for contour in cnts:
        if cv2.contourArea(contour) <5000:
            continue

        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        winsound.Beep(500, 200)

    status_list.append(status)

    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    #time.sleep(3)

    cv2.imshow('frame', frame)
    #cv2.imshow('Capturing', gray)
    #cv2.imshow('Canny', edged)
    #cv2.imshow('delta', delta_frame)
    #cv2.imshow('thresh', thresh_delta)
    #cv2.imshow('MOG2noise', fgmask)
    cv2.imshow('MOG2', fgmask0)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

#print(a)
#print(status_list)
#print(times)

for i in range(0, len(times), 2):
    #print(i)
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index= True)

df.to_csv("Times.csv")



video.release()
cv2.destroyAllWindows()
