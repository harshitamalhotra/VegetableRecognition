import cv2
import numpy as np
import os

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

print("Loading Images...")

Path, fol, datalength = readPath(path='newvegetables')

FruitArray = []
for i in range(2,len(Path)):
    fileName = Path[i]
    fruit = cv2.imread(fileName)
    fruit = cv2.resize(fruit, (100, 100))
    gray = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
    FruitArray.append(gray)
FruitArray = np.asarray(FruitArray)
FruitArray = FruitArray.reshape(FruitArray.shape[0], -1)

FruitArray = FruitArray/255.

print("Images Loaded Successfully...")

np.save("vegetables.npy", FruitArray)