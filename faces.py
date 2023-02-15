import numpy as np
import cv2
import faces_train
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascades\haarcascades_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades\haarcascades_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = faces_train.folder_ids
labels = {v:k for k,v in labels.items()}

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(frame,scaleFactor=1.2,minNeighbors=1)

    for x,y,w,h in face:
       
        region = frame[y:y+h,x:x+w]
        region_gray = gray[y:y+h,x:x+w]

        id, confidence = recognizer.predict(region_gray)
        if confidence >=45:
            print(id,confidence,labels[id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (0,255,0)
            stroke = 4
            cv2.putText(frame,name ,(x,y) ,font,1,color,stroke)
        color = (255,0,255)
        x_cordinates, y_cordinates = x+w, y+h
        cv2.rectangle(frame, (x,y), (x_cordinates,y_cordinates), color,3)
        # image = 'my_image.png'
        # cv2.imwrite(image,region)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF ==  ord('q'):
        break 
    

cap.release()
cap.destroyAllWindows()