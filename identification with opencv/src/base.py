import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    
    cv.imshow('Kamera', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
    

cap.release()
cv.destroyAllWindows()