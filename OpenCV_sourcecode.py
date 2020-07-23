#Import required libraries and packages into the coding environment

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

#Take a bounding predicted by dlib and convert it to the format (x, y, w, h) as we would normally do with OpenCV. Return a tuple of (x, y, w, h)
 
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
 
#Initialize the list of (x, y)-coordinates. Loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates. Return the list of (x, y)-coordinates
 
def shape_to_np(shape, dtype="int"):
     coords = np.zeros((68, 2), dtype=dtype)
     for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
      return coords

#Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor.
 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/comp 4060/DDD/shape_predictor_68_face_landmarks.dat')


#Determine the facial landmarks for the face region, then convert the facial landmark (x, y)- coordinates to a NumPy array. Convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box. Loop over the (x, y)-coordinates for the facial landmarks and draw them on the image. Show the output image with the face detections + facial landmarks
   
vid = cv2.VideoCapture(0)
count = 0
count_yawn = 0
previous_output = "ALERT"
mouth_output = "GOOD"

while(True):
    ret,image = vid.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye_f = (abs(shape[37,1]-shape[41,1])+abs(shape[38,1]-shape[40,1]))/abs(shape[36,0]-shape[39,0])
        right_eye_f = (abs(shape[43,1]-shape[47,1])+abs(shape[44,1]-shape[46,1]))/abs(shape[42,0]-shape[45,0])
      Yawn = (abs(shape[49,1]-shape[59,1])+abs(shape[61,1]-shape[67,1])+abs(shape[62,0]-shape[66,0])
             +abs(shape[63,0]-shape[65,0])+abs(shape[53,0]-shape[55,0]))/abs(shape[48,0]-shape[54,0])
   
         (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 127, 127), 1)
        
        if ((left_eye_f+right_eye_f)/2)>0.56:
            count = 0
            previous_output = "ALERT"
            cv2.putText(image,"ALERT", (x-10,y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        elif ((left_eye_f+right_eye_f)/2)<0.55:
            if previous_output == "DROWSY":
                count = count + 1
                cv2.putText(image,"DROWSY", (x-10,y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            if count>=30:
                count = count +1
                previous_output = "DROWSY alarm"
                cv2.putText(image,"DROWSY alarm", (x-10,y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            elif previous_output == "ALERT":
                previous_output = "DROWSY"
                count = 1
                cv2.putText(image,"DROWSY", (x-10,y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        cv2.putText(image,"count = %d" % count, (x + w + 20, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               
        if Yawn < 0.8:
            count_yawn = 0
            mouth_output = "GOOD"
            cv2.putText(image,"GOOD", (x-60,y-60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
        elif Yawn >= 0.8:
            mouth_output = "SLEEPY"
            count_yawn = count_yawn + 1
            
        if count_yawn >= 8:
            mouth_output = "SLEEPY"
            count_yawn = count_yawn+1
            cv2.putText(image," You are Sleepy.Please drink Water", (x-60,y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
       
        cv2.putText(image,"count_yawn = %d" % count_yawn, (x + w , y + h ),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       
                for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
