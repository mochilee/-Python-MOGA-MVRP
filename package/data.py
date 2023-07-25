# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:29:58 2022

@author: tony
"""
import numpy as np
import random as rd
import imag_resize as image


xs = [8, 50, 18, 35, 90, 40, 84, 74, 34, 40, 60, 74]
ys = [3, 62, 0, 25, 89, 71, 7, 29, 45, 65, 69, 47]
cities = ['Z', 'P', 'A', 'K', 'O', 'Y', 'N', 'X', 'G', 'Q', 'S', 'J']


def v_safe():
    v_r = np.empty((100,100))
    for i in range(0,100):
        for j in range(0,100):
            v_r[i][j] = round(rd.uniform(0.5,0.9),1)
    return v_r
def v_maze():
    v_m = np.zeros(shape=(100,100))
    for i in range(4):
        for t in range(len(v_m[0])):
            v_m[t][7+i] = 1
    for i in range(2):
        for t in range(len(v_m[0])):
            v_m[t][30+i] = 1
    for i in range(2):
        for t in range(len(v_m[0])):
            v_m[t][43+i] = 1
    for i in range(2):
        for t in range(len(v_m[0])):
            v_m[t][63+i] = 1
    for i in range(2):
        for t in range(50):
            v_m[t][76+i] = 1
            
            
    for i in range(5):
        for t in range(len(v_m[0])):
            v_m[1 + i][t] = 1
    for i in range(2):
        for t in range(87):
            v_m[16 + i][t] = 1
    for i in range(2):
        for t in range(85):
            v_m[31 + i][t] = 1
    for i in range(2):
        for t in range(60):
            v_m[56 + i][t] = 1
            
    for i in range(2):
        for t in range(54):
            v_m[45 + i][27 + t] = 1
    for i in range(2):
        for t in range(63):
            v_m[74 + i][17 + t] = 1
    return v_m

def v_Dmaze():
    d_maze = np.zeros(shape=(100,100))
    for i in range(7):
        for t in range(9):
            d_maze[6 + t][i] = 1
    return d_maze
def create_v():
    v_s = np.empty((100,100))
    v_r = np.empty((100,100))
    v_m = v_maze()
    
    v_s,v_r= image.create_vs_vr()
    return v_s,v_r,v_m

def create_v2():
    v_s = np.empty((100,100))
    v_r = np.empty((100,100))
    v_m = v_maze()
    d_maze = v_Dmaze()
    v_s,v_r= image.create_vs_vr()
    return v_s,v_r,v_m,d_maze

if  __name__ == '__main__':
    v_s,v_r,v_m,d_maze= create_v()
       
