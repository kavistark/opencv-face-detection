import cv2

facecascade = cv2.CascadeClassifier('reso/haarcascade_frontalface_default.xml')
img = cv2.imread('reso/face.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = facecascade.detectMultiScale(img_gray,1.1,4)

for (x,y,w,h) in faces :
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)

cv2.imshow('output',img)
cv2.waitKey(0)