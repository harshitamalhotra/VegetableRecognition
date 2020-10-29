import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("WELCOME TO VEGETABLE RECOGNITION")

def readPath(path):
    imgPath = []
    length = []
    Folder = []
    for root, folder, files in os.walk(path):
        for i in folder:
            Folder.append(i)
        # print(files)
        if len(files) <= 1:
            continue
        length.append(len(files))
        for file in files:
            if '._' in file:
                continue
            imgPath.append(root + '/' + file)
    return imgPath, Folder, length

Path, fol, datalength = readPath(path='newvegetables')
print("Loading Images...")
FruitArray = np.load('vegetables.npy')
print("Images Loaded Successfully...")

labels = np.zeros((FruitArray.shape[0], 1))
x = 0
y = 0
datalength.insert(0, 0)
print(datalength)
for j in range(len(datalength)-1):
    x += datalength[j]
    y += datalength[j + 1]
    labels[x:y] = float(j)

print("Training Started")


svm = SVC()
svm.fit(FruitArray,labels)

y_pred = svm.predict(FruitArray)
score = accuracy_score(y_pred,labels)

print("ACCURACY SCORE IS...")
print(score)
print("VEGETABLE AFTER PREDICTION...:")

img = cv2.imread('15.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
newimg = cv2.resize(gray, (100, 100))
newimg=newimg/255.
newimg = newimg.flatten()
image_pred = svm.predict([newimg])
userName = {}
for i in range(len(fol)):
    name = fol[i]
    userName[i] = name
name = userName[int(image_pred)]
print(name)

