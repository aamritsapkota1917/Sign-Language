import numpy as np
import cv2 as cv
from pathlib import Path

def get_image():
    Class = 'iloveU'
    Path('DATASET/'+Class).mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 1501    
    while True:
       
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv.flip(frame,1)
        i+= 1
        if i % 1==0:
            cv.imwrite('DATASET/'+Class+'/'+str(i)+'.png',frame)
      
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q') or i > 2000:
            break
  
    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
   get_image()
  