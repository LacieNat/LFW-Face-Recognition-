import numpy as np
import png
import os

import argparse
import cv2
import random
import shutil

import openface
import openface.helper
from openface.data import iterImgs

from subprocess import call

X_train = np.load('X_train.npy').reshape(-1,50, 37).astype(np.uint8)
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy').reshape(-1,50, 37).astype(np.uint8)

def testArrToImg():
    for i in range(len(X_test)):
        png.from_array(X_test[i], 'L').save("test/"+str(i)+".png")

def trainArrToImg():
    count0=0
    count1=0
    count2=0
    count3=0
    count4=0
    count5=0
    count6=0
    for i in range(len(X_train)):
        f=0
        c=0
        if y_train[i]==0:
    	    f=0
    	    count0+=1
    	    c = count0
        elif y_train[i]==1:
    	    f=1
    	    count1+=1
    	    c = count1
        elif y_train[i]==2:
       	    f=2
       	    count2+=1
       	    c = count2
        elif y_train[i]==3:
    	    f=3
    	    count3+=1
    	    c = count3
        elif y_train[i]==4:
    	    f=4
    	    count4+=1
    	    c = count4
        elif y_train[i]==5:
            f=5
            count5+=1
            c = count5
        elif y_train[i]==6:
    	    f=6
    	    count6+=1
    	    c = count6
	else:
            print(i)
    
        png.from_array(X_train[i], 'L').save("raw/"+str(f)+"/"+str(c)+".png")

def makeImgDir():
    directory = "./raw"
    if not os.path.exists(directory):
        os.makedirs(directory)
    subfolders = ["/0", "/1", "/2", "/3", "/4", "/5", "/6"]
    for s in subfolders:
        if not os.path.exists(directory+s):
            os.makedirs(directory+s)

    testDir = "./test"
    if not os.path.exists(testDir):
        os.makedirs(testDir)

makeImgDir()
trainArrToImg()
testArrToImg()

call(["./align-dlib.py", "./raw", "align", "outerEyesAndNose" , "./aligned", "--size", "96"])
call(["./main.lua", "-outDir" ,"./features", "-data" ,"./aligned"])
call(["./classifier.py", "train", "./features"])
call(["python", "cnn.py"])
c = "./classifier.py infer ./features/classifier.pkl ./test/*"
call([c], shell=True)

