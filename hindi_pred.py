import cv2
from keras.models import load_model
import numpy as np
from collections import deque

image_x=31
image_y=33
b=True
model=load_model("DevanagriScript.h5")

letters={0:'CHECK',1:'KA',2:'KHA',3:'GA',4:'GHA',5:'KNA',6:'SHUNYA',7:'EK',8:'DO',9:'TEEN',10:'CHAAR'}


def keras_predict(model,image):
    processed=keras_process_image(image)
    pred_probability=model.predict(processed)[0]
    pred_class=list(pred_probability).index(max(pred_probability))
    return max(pred_probability),pred_class

def keras_process_image(img):
    image_x=31
    image_y=33
    img=cv2.resize(img,(image_x,image_y))
    img=np.array(img,dtype=np.float32)
    img=np.reshape(img,(-1,image_x,image_y,1))
    return img

cap=cv2.VideoCapture(0)
lower_blue=np.array([110,50,50])
upper_blue=np.array([130,255,255])
pred_class=0
pts=deque(maxlen=512)
blackboard=np.zeros((480,640,3),dtype=np.uint8)
digit=np.zeros((200,200,3),dtype=np.uint8)

while cap.isOpened():
    ret,img=cap.read()
    img=cv2.flip(img,1)
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHSV,lower_blue,upper_blue)
    blur=cv2.medianBlur(mask,15)
    blur=cv2.GaussianBlur(blur,(5,5),0)
    thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cnts=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
    center=None
    if(len(cnts))>=1:
        contour=max(cnts,key=cv2.contourArea)
        if(cv2.contourArea(contour)>250):
            ((x,y),radius)=cv2.minEnclosingCircle(contour)
            cv2.circle(img,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(img,center,5,(0,255,255),-1)
            M=cv2.moments(contour)
            center=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
            pts.appendleft(center)
            for i in range(1,len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard,pts[i-1],pts[i],(255,255,255),2)
                cv2.line(img,pts[i-1],pts[i],(0,0,255),5)
    elif(len(cnts)==0):
        if(len(pts)!=[]):
           blackboard_gray=cv2.cvtColor(blackboard,cv2.COLOR_BGR2GRAY)
           blur1=cv2.medianBlur(blackboard,15)
           blur1=cv2.GaussianBlur(blur1,(5,5),0)
           
           thresh1=cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
           blackboard_cnts=cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
           if(len(blackboard_cnts)>=1):
               cnt=max(blackboard_cnts,key=cv2.contourArea)
               print(cv2.contourArea(cnt))
               if cv2.contourArea(cnt)>2000:
                   x,y,w,h=cv2.boundingRect(cnt)
                   digit=blackboard_gray[y:y+h,x:x+w]
                   pred_probab,pred_class=keras_predict(model,digit)
                   print(pred_class,pred_probab)
        pts=deque(maxlen=512)
        blackboard=np.zeros((480,640,3),dtype=np.uint8)
    cv2.putText(img,"Conv Network :"+str(letters[pred_class]),(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow("Frame",img)
    #cv2.imshow("contours",thresh)
    k=cv2.waitKey(10)
    if k==27:
      break
cap.release()
cv2.destroyAllWindows()
    
  
