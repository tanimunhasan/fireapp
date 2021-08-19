import cv2
#from playsound import playsound


fireDetect=cv2.CascadeClassifier('fire_detection.xml')

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fires=fireDetect.detectMultiScale(frame, 1.2, 5)
        name = "fire"
        for (x,y,w,h) in fires:
            #x1,y1=x+w, y+h
            cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.putText(frame,name,(x-20,y-20),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)
            print("fire is detected")
            cv2.imshow('frame', frame)
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 1)
            # cv2.line(frame, (x,y), (x+30, y),(255,0,255), 6) #Top Left
            # cv2.line(frame, (x,y), (x, y+30),(255,0,255), 6)
            #
            # cv2.line(frame, (x1,y), (x1-30, y),(255,0,255), 6) #Top Right
            # cv2.line(frame, (x1,y), (x1, y+30),(255,0,255), 6)
            #
            # cv2.line(frame, (x,y1), (x+30, y1),(255,0,255), 6) #Bottom Left
            # cv2.line(frame, (x,y1), (x, y1-30),(255,0,255), 6)
            #
            # cv2.line(frame, (x1,y1), (x1-30, y1),(255,0,255), 6) #Bottom right
            # cv2.line(frame, (x1,y1), (x1, y1-30),(255,0,255), 6)
            #playsound('audio.mp3')
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
