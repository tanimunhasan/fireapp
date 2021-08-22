#Developed by Tanimun Hasan
#import numpy as np
import cv2
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')
video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    fire0 = fire_cascade.detectMultiScale(frame0, 1.2, 5)
    fire1 = fire_cascade.detectMultiScale(frame1, 1.2, 5)
    name="fire"
    if (ret0):
        for (x,y,w,h) in fire0:
            cv2.rectangle(frame0, (x-20,y-20),(x+w+20,y+h+20),(255,0,0),4)
            roi_gray = gray0[y:y+h, x:x+w]
            roi_color = frame0[y:y+h, x:x+w]
            cv2.putText(frame0,name,(x-20,y-20),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)
        # Display the resulting frame
        cv2.imshow('Cam 0', frame0)

    if (ret1):
        for (x,y,w,h) in fire1:
                cv2.rectangle(frame1, (x-20,y-20),(x+w+20,y+h+20),(255,0,0),4)
                roi_gray = gray1[y:y+h, x:x+w]
                roi_color = frame1[y:y+h, x:x+w]
                cv2.putText(frame1,name,(x-20,y-20),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)
        # Display the resulting frame
        cv2.imshow('Cam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()
