import cv2
from numpy import *
import numpy as np

faceList = []
imgNum = 9
height = width = 100
eigenFaceNumber = 6
temp = sum = mat(zeros((1,height*width)))
deltaFace = mat(zeros((height*width,imgNum)))

for i in range(1,imgNum+1):                             #load images
    imgPath = "/Users/Silibett/Python project/FaceRecognition/att_faces/未命名文件夹/"+str(i)+".jpg"
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)          #convert image into gray
    img = cv2.resize(img,(width,height))                #standardize the size
    img = img.reshape((1,height*width))                 #n*n to 1*n^2
    sum = sum+img
    faceList.append(img)

unknownImagePath = "/Users/Silibett/Python project/FaceRecognition/att_faces/未命名文件夹/unknown2.jpg"
unknownImage = cv2.imread(unknownImagePath)
unknownImage = cv2.cvtColor(unknownImage,cv2.COLOR_RGB2GRAY)
unknownImage = cv2.resize(unknownImage,(width,height))
unknownImage = unknownImage.reshape((height*width),1)

average = sum//imgNum                   #average face
averageFace = average.reshape((width,height))
cv2.imwrite("average.jpg",averageFace)
for i in range(imgNum):
    temp = faceList[i]
    deltaFace[:,i] = (temp-average).T

C = dot(deltaFace.T,deltaFace)

eigenValues,eigenVectors = linalg.eig(C)        #calculate the eigenvalues and eigenvectors

index = np.argsort(-eigenValues)                #sorting
eigenValues = eigenValues[index]
eigenVectors = eigenVectors[:,index]

eigenFace = np.dot(deltaFace,eigenVectors)         #build eigenfaces

for i in range(eigenFaceNumber):
    temp = eigenFace[:,i].reshape((height,width))
    cv2.imwrite("EigenFace"+str(i)+".jpg",temp)

knownFaceNum = 5
knownFaceList = []

for i in range(1,knownFaceNum+1):                             #load images
    imgPath = "/Users/Silibett/Python project/FaceRecognition/att_faces/s12/"+str(i)+".jpg"
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)          #convert image into gray
    img = cv2.resize(img,(width,height))                #standardize the size
    img = img.reshape((1,height*width))                 #n*n to 1*n^2
    knownFaceList.append(img)

cor = mat(zeros((imgNum,eigenFaceNumber)))
average = average.T
for i in range(knownFaceNum):
    temp = knownFaceList[i].reshape((height * width, 1))
    for j in range(eigenFaceNumber):
        num = dot(eigenFace[:,j].T,temp-average)
        cor[i,j] = num
unknownCor = mat(zeros((1,eigenFaceNumber)))

for j in range(eigenFaceNumber):
    num = np.dot(eigenFace[:,j].T,unknownImage-average)
    unknownCor[0,j] = num

for i in range(knownFaceNum):
    distance = np.linalg.norm(unknownCor-cor[i])
    print(distance/1000)