import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Read the input image
img = cv2.imread('test.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3,5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y) , (x+w,y+h), (255,0,0),3)

    roi_gray=gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1,22)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey) , (ex+ew,ey+eh), (0,255,0),2)

    smiles = smile_cascade.detectMultiScale(roi_gray,1.7,23)

    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx,sy) , (sx+sw,sy+sh), (0,0,255),2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()



