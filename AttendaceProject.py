import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# creating a list that will get the images from our folder automatically
# and the generate the encoding for it automatically
# and then it will try to find it in our webcam

path = 'ImageAttendance'
#we will tell ask our program to find this folder ,
# the number of images in it and find the encoding for them
images = []
classNames = []
myList = os.listdir(path)#grabing the images of the ImageAttendance folder
print(myList)

#reading our current image of the class
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#function to find the encodings
def findEncodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

#function to generate the attendance
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('encoding completed') #ensuring encoding is done completely

#initializing webcam
cap=cv2.VideoCapture(0)

#loop to get each frame one by one
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    # to speeding up the process we're resizing the captured images
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    #finding encoding of the webcam images
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    #finding the matches

    #basically one by one it grabs one face location from facesCurFrame list
    #and then it grabs the encoding of encode face from encodeCurFrame list
    #and to have them in same loop we're using zip
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 #getting back original image size
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) #drawing a rectangale over our image
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)#another rectangle on our original image
            # putting the name in the bottom
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)



    cv2.imshow('Webcam', img)
    cv2.waitKey(1)