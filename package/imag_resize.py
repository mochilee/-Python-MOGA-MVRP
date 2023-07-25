import cv2 as cv
import os
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import copy 
def showImage(img):
    cv.namedWindow('img', 0)
    cv.resizeWindow('img', 600,400)
    cv.imshow("img", img)
    cv.waitKey()
 
def changeImage2(img):
    #pra为缩放的倍率
    image = cv.resize(img, (100, 100), interpolation=cv.INTER_AREA)
    #cv.imshow('Result', image)
    #cv.waitKey(0)
    return image
 
def changeImage(img, pra):
    #pra为缩放的倍率
    height, width = img.shape[:2]
    #此处要做integer强转,因为.resize接收的参数为形成新图像的长宽像素点个数
    size = (int(height*pra), int(width*pra))
    img_new = cv.resize(img, size, interpolation=cv.INTER_AREA)
    return img_new
def create_vs_vr():
    img = cv.imread('city5_test.png',1)
    #showImage(img)
    print('原图的shape',img.size)
#缩放倍率为0.5,即将原图的长宽都减少一半
    test = changeImage(img, 0.1)
    test2 = changeImage2(img)
    image = cv.cvtColor(test2, cv.COLOR_BGR2GRAY)
    #showImage(image)
    print('缩放后的shape',image.size)
    test2 = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):        
            if ((image[i][j]) in test2) == False:         
                test2.append(image[i][j])
    
    v_s = np.empty((100,100))
    v_r = np.empty((100,100))
    #v_s = np.empty((1000,1000))
    #v_r = np.empty((1000,1000))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):        
            if (image[i][j]) > 164:
                v_s[i][j]= round(rd.uniform(0.1,0.2),1)
                
            elif 130 <= (image[i][j]) <= 164:
                v_s[i][j]= round(rd.uniform(0.3,0.5),1)
    
            elif 100 <=  (image[i][j]) < 130:
                v_s[i][j]= round(rd.uniform(0.5,0.7),1)
    
            elif 90 <=  (image[i][j]) < 100:
                v_s[i][j]=round(rd.uniform(0.4,0.6),1)
    
            elif 60 <=  (image[i][j]) < 90:
               v_s[i][j]=round(rd.uniform(0.7,0.8),1)
    
            elif 25 <=  (image[i][j]) < 60:
                v_s[i][j]=(0.9)
    
            elif 0 <=  (image[i][j]) < 25:
                v_s[i][j]= (0.1)
                
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):        
            if (image[i][j]) > 164:
                v_r[i][j]= round(rd.uniform(0.1,0.5),1)
            elif 150 <= (image[i][j]) <=180:
                v_r[i][j]= 1

            elif 100 <=  (image[i][j]) < 180:
                v_r[i][j]= 1
 
            elif 90 <=  (image[i][j]) < 100:
                v_r[i][j]= 1
  
            elif 60 <=  (image[i][j]) < 90:
               v_r[i][j]=1
 
            elif 25 <=  (image[i][j]) < 60:
                v_r[i][j]=1

            elif 0 <=  (image[i][j]) < 25:
                v_r[i][j]= (1)
    
    
    return v_s,v_r


if  __name__ == '__main__':
    v_s,v_r= create_vs_vr()
    xs = [2, 5, 9, 13, 74, 32, 34, 35, 36, 37,
          58, 57, 58, 58, 59, 75, 78, 77, 21, 24,       
          25, 27, 46, 47, 48, 5, 91, 92, 93, 88]       
           #x座標
    ys = [8, 31, 43, 62, 17, 7, 29, 42, 60, 72,      
          4, 18, 26, 42, 59, 26, 58, 41, 36, 52,
          67,79, 35, 51, 71, 62, 16, 41, 58, 1] #y座標
    for i in range(len(xs)):
        if (v_r[xs[i]][ys[i]]) == 1:
            print(xs[i],ys[i])
