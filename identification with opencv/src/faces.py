
import numpy as np
import pickle
import cv2 as cv

face_cascades = cv.CascadeClassifier('src/data/cascades/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv.CascadeClassifier('src/data/cascades/haarcascade_eye.xml')
#smile_cascade = cv.CascadeClassifier('src/data/cascades/haarcascade_smile.xml')

recognizer = cv.face.LBPHFaceRecognizer.create()
recognizer.read("train.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors=4)
    for (x,y,w,h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]#(cord1-height, cord2-height)
        roi_color = frame[y:y+h, x:x+w]
        
        #recognize deep learned model predict keras tensorflow pytorch sckit learn
        
        #face
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_) 
            print(labels[id_])
            font =cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke =2
            cv.putText(frame , name, (x, y), font, 1, color, stroke, cv.LINE_AA)
             
        
        img_item = "my-image.png"
        cv.imwrite(img_item, roi_gray)
        
        color = (0, 255, 0)
        stroke = 2
        width = x + w
        height = y + h
        cv.rectangle(frame, (x, y), (width, height), color, stroke)
        #Eyes
        '''eyes =eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        smile = smile_cascade.detectMultiScale(roi_gray)
        for(sx, sy, sw, sh) in smile:
            cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,255,0), 2)'''
        
        
    cv.imshow('Kamera', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
    


cap.release()
cv.destroyAllWindows()