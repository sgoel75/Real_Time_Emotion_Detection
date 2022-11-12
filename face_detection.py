import cv2
import numpy as np
from model import FacialExpressionModel


cap = cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,640)
model=FacialExpressionModel("model.json","model_filter.h5")
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
while cap.isOpened():
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_flip = cv2.flip(gray_frame, 1)
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)



    for (x, y, w, h) in faces:
        face=gray_frame[y:y+h,x:x+w]
        roi=cv2.resize(face,(48,48))
        roi=np.reshape(roi,(48,48,1))
        roi=np.expand_dims(roi,axis=0)
        pred = model.predict_emotion(roi)
        cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
