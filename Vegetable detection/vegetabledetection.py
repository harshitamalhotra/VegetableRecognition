import cv2
import numpy as np
import os

#font = cv2.FONT_HERSHEY_COMPLEX


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
            imgPath.append(root + '/' + file)
    return imgPath, Folder, length


Path, fol, datalength = readPath(path='newvegetables')
# print(Path)
# print(fol)
# print(datalength)
userName = {}
for i in range(len(fol)):
    name = fol[i]
    userName[i] = name
print(userName)
print(datalength)
# THIS IS CONVERTING IMAGE
FruitArray = []
for i in range(len(Path)):
    fruit = cv2.imread(Path[i])
    fruit = cv2.resize(fruit, (100, 100))
    gray = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
    FruitArray.append(gray)
FruitArray = np.asarray(FruitArray)
FruitArray = FruitArray.reshape(FruitArray.shape[0], -1)
# print(FruitArray.shape)

# IMAGE CONVERTED
labels = np.zeros((FruitArray.shape[0], 1))
x = 0
y = 0
# WE DID INSERTING TO START SLICING FROM 0
datalength.insert(0, 0)
print(datalength)
try:
    for j in range(len(datalength)):
        x += datalength[j]
        y += datalength[j + 1]
        labels[x:y] = int(j)
        print(labels)
except BaseException:
    pass


def distance(x2, x1):
    return np.sqrt(sum((x1 - x2) ** 2))


def knn(x, train, k=5):
    n = train.shape[0]  # we took x or 100 (means 100 faces)
    d = []
    for i in range(n):
        d.append(distance(x, train[i]))
    d = np.asarray(d)
    indexes = np.argsort(d)  # gives index of the distances
    sortedLabels = labels[indexes][:k]
    count = np.unique(sortedLabels, return_counts=True)
    print(count)
    return count[0][np.argmax(count[1])]


Newimage = cv2.imread('15.jpg')
Newimage = cv2.cvtColor(Newimage, cv2.COLOR_BGR2GRAY)
Newimage = cv2.resize(Newimage, (100, 100))
label = knn(Newimage.flatten(), FruitArray)
name = userName[int(label)]
print(name)

