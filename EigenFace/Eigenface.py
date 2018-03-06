import cv2
from numpy import *
import numpy as np

faceList = []
imgNum = 36
height = width = 50
answerFaceNumber = 6
temp = sum = mat(zeros((1,height*width)))
deltaFace = mat(zeros((height*width,imgNum)))

for i in range(1,imgNum+1):                             #load images
    imgPath = "/Users/Silibett/Python project/EigenFace/faces/face"+str(i)+".jpg"
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)          #convert image into gray
    img = cv2.resize(img,(width,height))                #standardize the size
    img = img.reshape((1,height*width))                 #n*n to 1*n^2
    sum = sum+img
    faceList.append(img)

average = sum//imgNum                   #average face

for i in range(imgNum):
    temp = faceList[i]
    deltaFace[:,i] = (temp-average).T

C = dot(deltaFace.T,deltaFace)

eigenValues,eigenVectors = linalg.eig(C)        #calculate the eigenvalues and eigenvectors

index = np.argsort(-eigenValues)                #sorting
eigenValues = eigenValues[index]
eigenVectors = eigenVectors[:,index]


eigenFace = dot(deltaFace,eigenVectors)         #build eigenfaces

for i in range(answerFaceNumber):
    temp = eigenFace[:,i].reshape((height,width))
    cv2.imwrite("EigenFace"+str(i)+".jpg",temp)
