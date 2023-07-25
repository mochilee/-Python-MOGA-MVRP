# -*- coding: utf-8 -*-

import random as rd
import copy
from matplotlib import pyplot as plt
import data as data
import math
import bidirectional_a_star as astar
import numpy as np
import time
import gc
V_Safe,V_R,v_maze = data.create_v()

#maze = data.maze
x = 100 #地圖大小 如果更改地圖大小 此必須修改
y = 100 #地圖大小 如果更改地圖大小 此必須修改
s = 60 #工具速率
s1 = 60 #警車基礎速率
s2 = 40 #無人警車基礎速率
all_best_f1 = []
all_best_f2 = []
all_best_f3 = []
third_pop = []
third_pop_test = []


class Location:
    def __init__(self, name, x, y):
        self.name = name
        #name是城市的名稱，用於我們最後輸出路徑的時候知道到底經過了哪些路徑
        self.loc = (x, y)
        #loc就是城市的xy座標（這邊我們就當成整個地圖是xy平面）

    def astar(self,location2,vehicle): #呼叫astar函式 回傳rx ry(為路徑所經過的各點)
        assert isinstance(location2, Location)#除錯 判斷location2是為跟Location型態一樣
        return astar.a_star(self.loc[0], self.loc[1], location2.loc[0], location2.loc[1],vehicle) #需變數(起點位置 , 終點位置 , 載具種類)

    def two_point_distance(self,location2,vehicle):
        assert isinstance(location2, Location)#除錯 判斷location2是為跟Location型態一樣
        rx = []
        ry = []
        #print(f"1 {self.loc[0],self.loc[1]}")
        #print(f"2 {location2.loc[0],location2.loc[1]}")
        
        t = (location2.loc[0] * 1) - (self.loc[0] * 1)  
        x = (self.loc[1] * 1) - (location2.loc[1] * 1)
        y = (location2.loc[0] * self.loc[1]) - (self.loc[0] * location2.loc[1])
        #print(f'x = {x} y = {y} t = {t}')
        
        #fin_x = x / t
        #fin_y = y / t
        #print(f'{x}x + {t}y = {y}')
        rx.append(self.loc[0])
        ry.append(self.loc[1])
        for i in range(abs(self.loc[0] - location2.loc[0])):
            if (self.loc[0] < location2.loc[0]):             
                temp = self.loc[0] + (i + 1)
                lx = temp
                ly = (y - (x * lx)) / t
                rx.append(lx)
                ry.append(math.ceil(ly))
            elif (self.loc[0] > location2.loc[0]): 
                temp = self.loc[0] - (i + 1)
                lx = temp
                ly = (y - (x * lx)) / t
                rx.append(lx)
                ry.append(math.ceil(ly))
        #print(rx)
        #print(ry)
        return  rx,ry
    
def export_var(file_name,var1 = [],var2 = [],var3 = [],var4 = [],var5 = []): #結束輸出各極限值 txt檔
    name = file_name + "_5.txt"
    file1 = open(name, "w") 
    
    if file_name == "best_file":
        b_f1_path,b_f1_name = output_path_name_(0,var1,0)
        bestF1 = [var1.length,var1.resilience,var1.safe_value,b_f1_path,b_f1_name]
        
        b_f2_path,b_f2_name = output_path_name_(0,var2,0)
        bestF2 = [var2.length,var2.resilience,var2.safe_value,b_f2_path,b_f2_name]
        
        b_f3_path,b_f3_name = output_path_name_(0,var3,0)
        bestF3 = [var3.length,var3.resilience,var3.safe_value,b_f3_path,b_f3_name]
        
        bestAll = [var4[0],var4[1],var4[2],var4[3],var4[4]]
  
        bestF1_txt = repr(bestF1)
        bestF2_txt = repr(bestF2)
        bestF3_txt = repr(bestF3)
        bestAll_txt = repr(bestAll)
        file1.write("bestF1 = " + bestF1_txt + "\n")
        file1.write("bestF2 = " + bestF2_txt + "\n")
        file1.write("bestF3 = " + bestF3_txt + "\n")
        file1.write("bestAll = " + bestAll_txt + "\n")
        
    elif file_name == "all_best_f1":  
        for i in range(len(var1)):
            path_loc, path_name = output_path_name_(i,var1,1)        
            allbestF1 = [var1[i].length,var1[i].resilience,var1[i].safe_value,path_loc,path_name]
            txt = repr(allbestF1)
            file1.write(txt + "\n")
            
    elif file_name == "all_best_f2": 
        for i in range(len(var1)):        
            path_loc, path_name = output_path_name_(i,var1,1)  
            allbestF2 = [var1[i].length,var1[i].resilience,var1[i].safe_value,path_loc,path_name]
            txt = repr(allbestF2)
            file1.write(txt + "\n")
    elif file_name == "all_best_f3": 
        for i in range(len(var1)):  
            path_loc, path_name = output_path_name_(i,var1,1)  
            allbestF3 = [var1[i].length,var1[i].resilience,var1[i].safe_value,path_loc,path_name]
            txt = repr(allbestF3)
            file1.write(txt + "\n")

    
    file1.close()
     
def output_path_name_(idx,now_path,check): #路徑轉化
    if check == 0:
        idx_path_loc = [(loc.loc[0], loc.loc[1]) for loc in now_path.path]
        idx_path_name = [(loc.name) for loc in now_path.path]
    else:    
        idx_path_loc = [(loc.loc[0], loc.loc[1]) for loc in now_path[idx].path]
        idx_path_name = [(loc.name) for loc in now_path[idx].path]
    return idx_path_loc,idx_path_name


def third_pop_cal(level): #計算第三族群，移除相同路徑，移除q不為0
    global all_best_f1,all_best_f2,all_best_f3
    global third_pop
    start = time.time()
    third_pop_temp = []
    
    new_third_pop = []
    
    
    new_third_pop.append(third_pop[0])
    for i in range(len(third_pop)):
        check = False
        new_third_pop.sort(key=lambda x: x.length, reverse=True)
        for j in new_third_pop:  
            
            if third_pop[i].length == j.length and third_pop[i].resilience == j.resilience and third_pop[i].safe_value == j.safe_value:
                check = True
                break
            else: 

                check = False
        #print(f'check {check}')
        
        if check is False:
           new_third_pop.append(third_pop[i])
    
    third_pop = new_third_pop
    
    
    pqc,q = fitness(third_pop)
    
    for i in range(len(third_pop)):
        third_pop[i].gpsiff = pqc[i]
        third_pop[i].q = q[i]
    idx = []
    for i in range(len(q)):
        if q[i] > 0:
            idx.append(i)
        
    count = 0
    for i in idx:
        third_pop.pop(i - count)
        count += 1
    '''
    for i in third_pop: 
        if i.q > 0:
            #print("=============del==========================================")
            #print(i.q)
            third_pop.pop(count)              
            #print("==========================================================")
        count += 1
    ''' 
    '''
    print("=============test==================")
    for i in third_pop:
        print(f'pqc :{i.gpsiff} safe_value: {i.safe_value} efficiency: {i.length} resilience: {i.resilience} Q = {i.q}')
    print("=============test==================")
    '''
    
    '''    
    for i in third_pop:
        third_pop_temp.append([i.length,i.resilience,i.safe_value])
    
    third_pop_test.append(third_pop_temp) 
    '''    
    
    if level % 10 == 0 :      
        f1_idx = 0
        f2_idx = 0
        f3_idx = 0        
        f1_temp = third_pop[0].length#預設值   
        f2_temp = third_pop[0].resilience #預設值   
        f3_temp = third_pop[0].safe_value #預設值
        
        for i in range(len(third_pop)):
            if f1_temp > third_pop[i].length:
                f1_temp = third_pop[i].length
                f1_idx = i
            if f2_temp > third_pop[i].resilience:
                f2_temp = third_pop[i].resilience 
                f2_idx = i
            if f3_temp < third_pop[i].safe_value:
                f3_temp = third_pop[i].safe_value    
                f3_idx = i
                
        all_best_f1.append(third_pop[f1_idx])       
        all_best_f2.append(third_pop[f2_idx]) 
        all_best_f3.append(third_pop[f3_idx])  
    end = time.time() 

    #print("===============================")  
    print(f'第三族群計算耗費時間 {np.round((end - start),2)}')
    print("===============================")   
    return third_pop        
    
def third_pop_lim(all_best_f1,all_best_f2,all_best_f3): #結束時計算第三族各F極限值

    best_f1_temp = all_best_f1[0].length
    best_f2_temp = all_best_f2[0].resilience
    best_f3_temp = all_best_f3[0].safe_value
    best_f1_idx = 0
    best_f2_idx = 0
    best_f3_idx = 0
    for i in range(len(all_best_f1)):
        if best_f1_temp > all_best_f1[i].length:
            best_f1_temp = all_best_f1[i].length
            best_f1_idx = i
        if best_f2_temp > all_best_f2[i].resilience:
            best_f2_temp = all_best_f2[i].resilience
            best_f2_idx = i
        if best_f3_temp < all_best_f3[i].safe_value:
            best_f3_temp = all_best_f3[i].safe_value
            best_f3_idx = i

           
    print(best_f1_idx,best_f2_idx,best_f3_idx)
    best_F1 = all_best_f1[best_f1_idx]
    best_F2 = all_best_f2[best_f2_idx]    
    best_F3 = all_best_f3[best_f3_idx]    
    return best_F1,best_F2,best_F3


#警車 無人警車起點(170,70)
#無人機 起點(760,410)
def plot_show(best_route): #打印路線
    v1 = []
    v2 = []
    sdv1 = []
    sdv2 = []
    d1=[]
    d2=[]
    d3=[]
    d4=[]
    d5=[]
    d6=[]
    rxy = []
    #ry = []
    zero_count = 0
    print("========================input======================")
    print([(loc.loc[0], loc.loc[1]) for loc in best_route])
    print([(loc.name) for loc in best_route])
    v1.append(Location(0, 17, 7)) #v1開始
    fig, ax = plt.subplots()
    for i in range(len(best_route)):  # 座標數值回調，原座標會因為突變交配等因素導致各起點有誤，此處根據各載具位置調整座標
        if best_route[i].name == "0": #檢測基因的0個數 檢測到n次代表載具更換(n自行調整)
            zero_count +=1 
        #正式測試 警車2台 無人警車2台 無人機6台
        #print(f"now {best_route[i].loc[0],best_route[i].loc[1]} zero = {zero_count}")
        if zero_count == 0: #count為1 表示目前為警車2 如要更改畫線載具數量需從此更改
            rxy.append(best_route[i].loc)
            v1.append(best_route[i])
               
        if zero_count == 1: #count為1 表示目前為警車2 如要更改畫線載具數量需從此更改
            if best_route[i].name == "0":
               best_route[i].loc = (17,7)
               v1.append(Location(0, 17, 7)) #v1結束

               rxy.append((17, 7))               
               v2.append(Location(0, 17, 7)) #v2開始
            else:                
               rxy.append(best_route[i].loc)               
               v2.append(best_route[i])

        elif zero_count == 2:#count為2 表示目前為無人警車1 如要更改畫線載具數量需從此更改
            if best_route[i].name == "0":
               best_route[i].loc = (17,7) 
               v2.append(Location(0, 17, 7)) #v2結束
               
               rxy.append((17, 7))               
               sdv1.append(Location(0, 17, 7)) #sdv1開始
            else:
               rxy.append(best_route[i].loc)               
               sdv1.append(best_route[i]) 
                   
        elif zero_count == 3:#count為3 表示目前為無人警車2 如要更改畫線載具數量需從此更改 
            if best_route[i].name == "0":
               best_route[i].loc = (17,7) 
               sdv1.append(Location(0, 17, 7)) #sdv1結束
               
               rxy.append((17, 7))               
               sdv2.append(Location(0, 17, 7)) #sdv2開始
            else:
               rxy.append(best_route[i].loc)               
               sdv2.append(best_route[i]) 

            
        elif zero_count == 4:#count為4 表示目前為無人車1 如要更改畫線載具數量需從此更改 
            if best_route[i].name == "0":
               best_route[i].loc = (76,41) 
               sdv2.append(Location(0, 17, 7)) #sdv2結束
               
               rxy.append((76, 41))               
               d1.append(Location(0, 76, 41)) #d1開始
            else:
               rxy.append(best_route[i].loc)               
               d1.append(best_route[i]) 
               
        elif zero_count == 5:#count為5 表示目前為無人車2 如要更改畫線載具數量需從此更改 
            if best_route[i].name == "0":
               best_route[i].loc = (17,7) 
               d1.append(Location(0, 76, 41)) #d1結束
               
               rxy.append((76, 41))               
               d2.append(Location(0, 76, 41)) #d2開始
            else:
               rxy.append(best_route[i].loc)               
               d2.append(best_route[i])
               
        elif zero_count == 6:#count為6 表示目前為無人車3 如要更改畫線載具數量需從此更改 
            if best_route[i].name == "0":
               best_route[i].loc = (76,41) 
               d2.append(Location(0, 76, 41)) #d2結束
               
               rxy.append((76, 41))               
               d3.append(Location(0, 76, 41)) #d3開始
            else:
               rxy.append(best_route[i].loc)               
               d3.append(best_route[i]) 
        elif zero_count == 7:#count為7 表示目前為無人車4 如要更改畫線載具數量需從此更改 
            if best_route[i].name == "0":
               best_route[i].loc = (76,41) 
               d3.append(Location(0, 76, 41)) #d3結束
               
               rxy.append((76, 41))               
               d4.append(Location(0, 76, 41)) #d4開始
            else:
               rxy.append(best_route[i].loc)               
               d4.append(best_route[i]) 
        elif zero_count == 8:#count為8 表示目前為無人車5 如要更改畫線載具數量需從此更改 
            if best_route[i].name == "0":
               best_route[i].loc = (76,41) 
               d4.append(Location(0, 76, 41)) #d4結束
               
               rxy.append((76, 41))               
               d5.append(Location(0, 76, 41)) #d5開始
            else:
               rxy.append(best_route[i].loc)               
               d5.append(best_route[i]) 
        elif zero_count == 9:#count為9 表示目前為無人車6 如要更改畫線載具數量需從此更改 
            if best_route[i].name == "0":
               best_route[i].loc = (76,41) 
               d5.append(Location(0, 76, 41)) #d5結束
               
               rxy.append((76, 41))               
               d6.append(Location(0, 76, 41)) #d6開始
            else:
               rxy.append(best_route[i].loc)               
               d6.append(best_route[i]) 
    d6.append(Location(0, 76, 41))
    print("========================output=========================")
    #print([(loc[0], loc[1]) for loc in rxy])
    #print([(loc.name) for loc in best_route])
    #畫線 一行代表一條線 此處有雖同載具但以不同線為表示
    '''
    print(f'v = {[loc.loc[0] for loc in v] ,  [loc.loc[1] for loc in v]}')
    print(f'sdv = {[loc.loc[0] for loc in sdv] ,  [loc.loc[1] for loc in sdv]}')
    print(f'd1 = {[loc.loc[0] for loc in d1] ,  [loc.loc[1] for loc in d1]}')
    print(f'd2 = {[loc.loc[0] for loc in d2] ,  [loc.loc[1] for loc in d2]}')
    '''
    ax.plot([loc.loc[0] for loc in v1], [loc.loc[1] for loc in v1], 'b', linestyle='dashed', marker='')
    ax.plot([loc.loc[0] for loc in v2], [loc.loc[1] for loc in v2], 'b', linestyle='dashed', marker='')
    ax.plot([loc.loc[0] for loc in sdv1], [loc.loc[1] for loc in sdv1], 'k', linestyle='dashed', marker='')
    ax.plot([loc.loc[0] for loc in sdv2], [loc.loc[1] for loc in sdv2], 'k', linestyle='dashed', marker='') 
    ax.plot([loc.loc[0] for loc in d1], [loc.loc[1] for loc in d1], 'y', linestyle='dashed', marker='')
    ax.plot([loc.loc[0] for loc in d2], [loc.loc[1] for loc in d2], 'y', linestyle='dashed', marker='')
    ax.plot([loc.loc[0] for loc in d3], [loc.loc[1] for loc in d3], 'm', linestyle='dashed', marker='')
    ax.plot([loc.loc[0] for loc in d4], [loc.loc[1] for loc in d4], 'm', linestyle='dashed', marker='')
    ax.plot([loc.loc[0] for loc in d5], [loc.loc[1] for loc in d5], 'r', linestyle='dashed', marker='')
    ax.plot([loc.loc[0] for loc in d6], [loc.loc[1] for loc in d6], 'r', linestyle='dashed', marker='')

    ax.scatter(xs, ys) #點散圖
    ax.grid(True) #表格
    for i in range(len(rxy)): #打印座標名 判斷時為更改座標名為各載具起點 目前為 警車,無人警車(5,5) 無人機(70,70)
        if (rxy[i][0] == 76 and rxy[i][1] == 41):
            txt = "D_S"           
        elif(rxy[i][0] == 17 and rxy[i][1] == 7):
            txt = "V_S"
        else:
            txt = best_route[i].name
        ax.annotate(txt, (rxy[i][0], rxy[i][1])) #文字註解
    #plt.show()
    #plt.close()
    plt.savefig("test.png") #儲存圖片
    
def create_locations(): #創建初始座標位置 (回傳座標 座標名)
    locations = []
    '''xs = [15, 54, 87, 132, 745, 320, 335, 351, 356, 368,
          567, 574, 578, 585, 594, 750, 775, 765, 212, 237,       
          256, 275, 467, 469, 478, 56, 906, 924, 935, 880]       
           #x座標
    ys = [81, 308, 431, 620, 170, 65, 286, 422, 598, 720,      
          40, 177, 264, 415, 586, 256, 584, 413, 360, 522, 670,
          792, 348, 510, 708, 624, 164, 405, 578, 13] #y座標'''
    
    xs = [2, 5, 9, 13, 74, 32, 34, 35, 36, 37,
          58, 57, 58, 58, 59, 75, 78, 77, 21, 24,       
          25, 27, 46, 47, 48, 5, 91, 92, 93, 88]       
           #x座標
    ys = [8, 31, 43, 62, 17, 7, 29, 42, 60, 72,      
          4, 18, 26, 42, 59, 26, 58, 41, 36, 52,
          67,79, 35, 51, 71, 62, 16, 41, 58, 1] #y座標
    cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J', 
              'K', 'L', 'M', 'N', 'O', 'P','Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X','Y', 'Z','Aa','Ab','Ac','Ad'] #座標名
    for x, y, name in zip(xs, ys, cities):
        locations.append(Location(name, x, y)) 
    return locations, xs, ys, cities


def fitness(routes): #gpsiff 適性度計算 (回傳 p - q + c)
   pqc = []
   q_array = []
   for i in range(len(routes)):
       p = 0
       q = 0
       for j in range(len(routes)):   # f1 為效率最高(數值越低) f2為應變能力最佳(數值越低) f3為安心指數最高(數值越高)   
          if(i != j):
              pf1 = routes[i].length <= routes[j].length
              pf2 = routes[i].resilience <= routes[j].resilience
              pf3 = routes[i].safe_value >= routes[j].safe_value
              pb = pf1 and pf2 and pf3
              

              bf1 = routes[i].length == routes[j].length
              bf2 = routes[i].resilience == routes[j].resilience
              bf3 = routes[i].safe_value == routes[j].safe_value             
              b = bf1 and bf2 and bf3

              
              qf1 = routes[i].length >= routes[j].length
              qf2 = routes[i].resilience >= routes[j].resilience
              qf3 = routes[i].safe_value <= routes[j].safe_value                             
              qb = qf1 and qf2 and qf3

              
              if (pb and not b):                     
                  p+=1 
                  #print(f"test 1 p = {p}")
              elif (qb and not b):
                  q+=1
                  #print(f"test 2 q = {q}")
            
       pqc.append(p - q + len(routes)) 
       q_array.append(q)       
       #print(f'{testset[i].f1,testset[i].f2,testset[i].f3} p = {p} q = {q}')
   #print(f'pqc = {pqc}')
   return pqc,q_array

def path_fix(path): #座標修復 原座標會因為突變交配等因素導致各起點非0，以方便管理此處根據將起點座標統一歸0，在計算效率時再進行各起點調整
    #回傳 路徑
    check_count = 0

    for i in range(len(path)): #根據0的數量 判別起點
        if path[i].name == '0':
            check_count += 1 

        if check_count >= 2 and path[i].name == '0':
            #print(f'D {path[i].loc} i = {i}')
            path[i].loc = (0,0)

            #print(f'D {path[i].loc} i = {i}')
        elif check_count <= 1 and path[i].name == '0':
            #print(f'SDV {path[i].loc} i = {i}')
            path[i].loc = (0,0)

    del check_count  
    return path
def congestion_cal(rx,ry,V_R,V): #需傳入 astar回傳的路徑座標,擁擠率矩陣,載具名稱
    #擁擠率矩陣 * 相對應車輛速率 再與1相除(1格假定為1公里) 算出答案單位為小時 (回傳當格所耗時間)
    congestion_sum = 0
    now_v_r = V_R[int(rx[0])][int(ry[0])] + V_R[int(rx[len(rx) - 1])][int(ry[len(rx) - 1])]
    if V == "V": #當載具為警車 基礎速率為60
        s = 60
        congestion_sum += np.round(len(rx) / (s * now_v_r),3)
        '''for i in range(len(rx)):
            congestion_sum += np.round(1 / (s * V_R[int(rx[i])][int(ry[i])]),3)'''
        #print(f"V_congestion_sum = {congestion_sum}")
    elif V == "SDV": #當載具為無人警車 基礎速率為40
        s = 50
        congestion_sum += np.round(len(rx) / (s * now_v_r),3)
        '''for i in range(len(rx)):
            congestion_sum += np.round(1 / (s * V_R[int(rx[i])][int(ry[i])]),3)'''
        #print(f"SDV_congestion_sum = {congestion_sum}")
    elif V == "D": ##當載具為無人機 基礎速率為30 但無人機不受壅擠率矩陣影響
        s = 30
        congestion_sum += np.round(len(rx) / s ,3)
        '''for i in range(len(rx)):
            congestion_sum += np.round(1 / s,3)'''
        #print(f"D_congestion_sum = {congestion_sum}")
    fin = np.round(congestion_sum,2)
    #print(f'fin {fin}')
    return fin
    
def path_input(path_array,rx,ry,veh): #f3 安全指數路徑範圍判定 需傳入該載具安全指數範圍矩陣,astar回傳的路徑座標,載具名稱 (回傳該載具安全指數範圍矩陣)

    M_row = 100 #該地圖最高長  ex. 地圖為100 * 100 最高長寬為100 最低為0 如果有更改地圖長寬需更改此處
    M_col = 100 #該地圖最高寬
    m_row = 0 #該地圖最低長
    m_col = 0 #該地圖最低寬
    
    for i in range(len(rx)): #根據傳入載具名稱將標示進行變更
        if veh == 'V': #如果為警車 標示為1
            mark = 1
        elif veh == 'D': #如果為無人機 標示為4
            mark = 4
        elif veh == 'SDV': #如果為無人警車 標示為4
            mark = 2
        else:
            print('error') 
            
        #根據astar回傳的路徑座標為基準，以該座標的九宮格判定為該載具的安全指數路徑範圍內(目前如果有重複不進行累加)   
        if (m_row < int(rx[i] - 1) < M_row) and (m_col < int(ry[i] + 1) < M_col):   # 左上
            if  path_array[int(rx[i] - 1)][int(ry[i] + 1)] == 0:
                path_array[int(rx[i] - 1)][int(ry[i] + 1)] += mark 
              
        if (m_row < int(rx[i] + 0) < M_row) and (m_col < int(ry[i] + 1) < M_col):   # 左中
            if  path_array[int(rx[i] + 0)][int(ry[i] + 1)] == 0:
                path_array[int(rx[i] + 0)][int(ry[i] + 1)] += mark
       
        if (m_row < int(rx[i] + 1) < M_row) and (m_col < int(ry[i] + 1) < M_col):   # 右上
            if  path_array[int(rx[i] + 1)][int(ry[i] + 1)] == 0:
                path_array[int(rx[i] + 1)][int(ry[i] + 1)] += mark
        
        if (m_row < int(rx[i] - 1) < M_row) and (m_col < int(ry[i] + 0) < M_col):   # 中左
            if  path_array[int(rx[i] - 1)][int(ry[i] + 0)] == 0:
                path_array[int(rx[i] - 1)][int(ry[i] + 0)] += mark       
        
        if (m_row < int(rx[i] + 0) < M_row) and (m_col < int(ry[i] + 0) < M_col):   # 中心
            if  path_array[int(rx[i] + 0)][int(ry[i] + 0)] == 0:
                path_array[int(rx[i] + 0)][int(ry[i] + 0)] += mark   
                       
        if (m_row < int(rx[i] + 1) < M_row) and (m_col < int(ry[i] + 0) < M_col):   # 中右
            if  path_array[int(rx[i] + 1)][int(ry[i] + 0)] == 0:
                path_array[int(rx[i] + 1)][int(ry[i] + 0)] += mark     
            
        if (m_row < int(rx[i] - 1) < M_row) and (m_col < int(ry[i] - 1) < M_col):   # 下左
            if  path_array[int(rx[i] - 1)][int(ry[i] - 1)] == 0:
                path_array[int(rx[i] - 1)][int(ry[i] - 1)] += mark     
 
        if (m_row < int(rx[i] + 0) < M_row) and (m_col < int(ry[i] - 1) < M_col):   # 下中
            if  path_array[int(rx[i] + 0)][int(ry[i] - 1)] == 0:
                path_array[int(rx[i] + 0)][int(ry[i] - 1)] += mark
 
        if (m_row < int(rx[i] + 1) < M_row) and (m_col < int(ry[i] - 1) < M_col):   # 下右
            if  path_array[int(rx[i] + 1)][int(ry[i] - 1)] == 0:
                path_array[int(rx[i] + 1)][int(ry[i] - 1)] += mark
        
    return path_array

def create_path_array(x,y): #創建空矩陣 需傳入地圖大小 ex地圖 100*100 x就為100 y為100 (回傳n*n 零矩陣)
    return np.zeros((x, y)) 
  
class Route:
    def __init__(self, path):
        # path is a list of Location obj
        
        #self.path = path
        self.path = path_fix(path) #方便管理在每次交配完的路徑都進行起點歸零化
        self.V_path_array = create_path_array(x,y) #創建警車的安全指數矩陣 供f3使用
        self.SDV_path_array = create_path_array(x,y)#創建無人警車的安全指數矩陣 供f3使用
        self.D_path_array = create_path_array(x,y)#創建無人機的安全指數矩陣 供f3使用
        
        #path 是一個list 裡面裝著一個個Location物件

        self.length,self.tol_array = self._set_length() #計算 f1 效率 (回傳效率,總安全指數矩陣)
        #self.astar_length = self._set_astar_length()
        #length 是這條路徑的長度
        self.resilience = self._set_resilience() #計算 f2 應變能力 (回傳應變能力)
        self.safe_value = self._set_safe_value(self.tol_array) #計算 f3 安全指數 (回傳安全指數
        self.gpsiff = 0 
        self.q = 0
        #path_names = [loc.name for loc in self.path]       
        #print(f'path : {path_names}')
        #print([(loc.loc[0], loc.loc[1]) for loc in self.path])
        #print(f'safe_value: {self.safe_value} efficiency: {self.length} resilience: {self.resilience} Q : {self.q}')
        

        


    
    def _set_length(self): #計算 f1 效率 (回傳效率,總安全指數矩陣)
        #V_arry = []
        #D_arry = []
        #SDV_arry = []
        end_check = True
        #_set_length 只是透過path來計算路徑長度的函式
        #Note: 這邊有用到一些像是self.path[:]或是copy等等，是因為python裡面的list是mutable，這樣做可以避免我們在計算路徑的過程中不小心pop到self.path
        total_length = 0
        zero_countdown = 0
        #for i in self.path:
           #print(i.name)
        v_total_length = 0    
        sdv_total_length = 0
        d_total_length = 0
        #print(f'=======') 
        
        '''
        print("========================_set_length test=================================")        
        print([loc.name for loc in self.path])
        print([(loc.loc[0], loc.loc[1]) for loc in self.path])
        print("=====================================================")
        '''
        path_copy = self.path[:] #複製一份路徑的備份以免在後面的變動中影響原數值

        #警車 無人警車起點(17,7)
        #無人機 起點(76,41)
        #2台警車  2台無人警車 6台無人機
        from_here = Location('start',17,7) #從哪開始，因原路徑無第一個起點，此處便自行加一個初始起點方便計算
        #V_arry.append((5,5)) #將警車第一個起點加入警車的路徑矩陣
        #print(from_here.name)
        
        while path_copy: #根據複製的路徑進行逐一計算
            
            to_there = path_copy.pop(0) #移除複製路徑中當下的第一個座標 並將設為終點
            
            #print(to_there.name)
            if(to_there.name == "0"):   #如果終點座標名為0表示一條載具路徑的結束        
                zero_countdown += 1 
                #print(f"pass zero next route \\ zero : {zero_countdown}")
                #print(f'now check {zero_countdown}  {end_check}')
                if zero_countdown == 2 or zero_countdown == 4: #根據車輛改變
                    end_check = False
                #print(f'after check {zero_countdown}  {end_check}')
                
                
            if(zero_countdown <= 2): #如果count為0表示目前還在警車的路徑中 ，如更改警車數量此處需改變
                #print("frist route")
                if(to_there.name == "0"): #如果當下終點座標名為0 表示上條路徑已結束 故不算上條起點到終點的距離
                    to_there.loc = (17,7) #起始點位置更改
                    #V_arry.append(to_there.loc) #將當下終點加入無人警車的路徑矩陣
                    
                    #rx,ry= to_there.astar(from_here,"V") #呼叫astar 回傳astar路徑座標
                    
                    rx,ry = to_there.two_point_distance(from_here, "V")
                    #print('=============V end==============')
                    #print(f'rx = {rx}')
                    #print(f'ry = {ry}')
                    #print('=============V end==============')
                    fin_time = congestion_cal(rx,ry,V_R,"V") #計算f1效率
                    v_total_length += fin_time #計算該載具總效率
    
                    from_here = copy.deepcopy(to_there) #將當下的終點設為下次的起點 以改值不改址的方式
                    self.V_path_array = path_input(self.V_path_array,rx,ry,'V') #呼叫計算安全指數範圍函式

                else:
                    #V_arry.append(to_there.loc)  #將當下終點加入警車的路徑矩陣
                    #rx,ry= to_there.astar(from_here,"V") #呼叫astar 回傳astar路徑座標
                    
                    rx,ry = to_there.two_point_distance(from_here, "V")
                    #print('=============V==============')
                    #print(f'from_here:{from_here.loc} to_there :{to_there.loc} ')
                    #print(f'rx = {rx}')
                    #print(f'ry = {ry}')
                    #print('=============V==============')
                    fin_time = congestion_cal(rx,ry,V_R,"V") #計算f1效率
                    v_total_length += fin_time #計算該載具總效率
    
                    from_here = copy.deepcopy(to_there) #將當下的終點設為下次的起點 以改值不改址的方式
                    self.V_path_array = path_input(self.V_path_array,rx,ry,'V') #呼叫計算安全指數範圍函式

            elif(zero_countdown <= 4 and zero_countdown > 2): #如果count為1表示目前還在無人警車的路徑中 ，如更改無人警車數量此處需改變
                #print("second route")
 
                if(to_there.name == "0"): #如果當下終點座標名為0 表示上條路徑已結束 故不算上條起點到終點的距離
                #計算切換無人警車路線時警車回原點
                  if end_check == False:
                      to_there.loc = (17,7) #起始點位置更改
                      #V_arry.append(to_there.loc) #將當下終點加入無人警車的路徑矩陣
                      #SDV_arry.append(to_there.loc) #將當下終點加入無人警車的路徑矩陣
     
                     #rx,ry= to_there.astar(from_here,"V") #呼叫astar 回傳astar路徑座標
                      rx,ry = to_there.two_point_distance(from_here, "V")
                      #print('=============V end==============')
                      #print(f'from_here:{from_here.loc} to_there :{to_there.loc} ')
                      #print(f'rx = {rx}')
                      #print(f'ry = {ry}')
                      #print('=============V end==============')
                      fin_time = congestion_cal(rx,ry,V_R,"V") #計算f1效率
                      v_total_length += fin_time #計算該載具總效率    
                      from_here = copy.deepcopy(to_there) #將當下的終點設為下次的起點 以改值不改址的方式
                      self.V_path_array = path_input(self.V_path_array,rx,ry,'V') #呼叫計算安全指數範圍函式
                      end_check = True
                      
                  else:
                      to_there.loc = (17,7) #起始點位置更改
                      #SDV_arry.append(to_there.loc) #將當下終點加入無人警車的路徑矩陣                     
                      #rx,ry= to_there.astar(from_here,"SDV")  #呼叫astar 回傳astar路徑座標
                      rx,ry = to_there.two_point_distance(from_here, "SDV")
                      #print('===========SDV END============')
                      #print(f'from_here:{from_here.loc} to_there :{to_there.loc} ')
                      #print(f'rx = {rx}')
                      #print(f'ry = {ry}')
                      #print('===========SDV end==============')
                      fin_time = congestion_cal(rx,ry,V_R,"SDV") #計算f1效率
                      sdv_total_length += fin_time #計算該載具總效率
                      from_here = copy.deepcopy(to_there)  #將當下的終點設為下次的起點 以改值不改址的方式
                      self.SDV_path_array = path_input(self.SDV_path_array,rx,ry,'SDV') #呼叫計算安全指數範圍函式

                else:  
                    #SDV_arry.append(to_there.loc) #將當下終點加入無人警車的路徑矩陣
                    #rx,ry= to_there.astar(from_here,"SDV")  #呼叫astar 回傳astar路徑座標
                    rx,ry = to_there.two_point_distance(from_here, "SDV")
                    #print('===========SDV============')
                    #print(f'from_here:{from_here.loc} to_there :{to_there.loc} ')
                    #print(f'rx = {rx}')
                    #print(f'ry = {ry}')
                    #print('===========SDV==============')
                    fin_time = congestion_cal(rx,ry,V_R,"SDV") #計算f1效率
                    sdv_total_length += fin_time #計算該載具總效率
                    from_here = copy.deepcopy(to_there)  #將當下的終點設為下次的起點 以改值不改址的方式
                    self.SDV_path_array = path_input(self.SDV_path_array,rx,ry,'SDV') #呼叫計算安全指數範圍函式
                
                
            elif(zero_countdown > 4):  #如果count為2以上表示目前還在無人機的路徑中 ，如更改無人機數量此處需改變
                #print("third route")
                if (to_there.name == "0"): #如果當下終點座標名為0 表示上條路徑已結束 故不算上條起點到終點的距離
                #計算切換無人機路線時 無人警車需回原點
                    if end_check == False:
                        to_there.loc = (17,7)
                        #SDV_arry.append(to_there.loc) #將當下終點加入無人警車的路徑矩陣                       
                        #rx,ry= to_there.astar(from_here,"SDV")  #呼叫astar 回傳astar路徑座標
                        rx,ry = to_there.two_point_distance(from_here, "SDV")
                        #print('===========SDV END============')
                        #print(f'from_here:{from_here.loc} to_there :{to_there.loc} ')
                        #print(f'rx = {rx}')
                        #print(f'ry = {ry}')
                        #print('===========SDV end==============')
                        fin_time = congestion_cal(rx,ry,V_R,"SDV") #計算f1效率
                        sdv_total_length += fin_time #計算該載具總效率
                        from_here = copy.deepcopy(to_there)  #將當下的終點設為下次的起點 以改值不改址的方式
                        from_here.loc = (76,41)  
                        #D_arry.append(from_here.loc) #將當下終點加入無人警車的路徑矩陣
                        self.SDV_path_array = path_input(self.SDV_path_array,rx,ry,'SDV') #呼叫計算安全指數範圍函式
                        end_check = True
                    else:
                        to_there.loc = (76,41) #起始點位置更改
                        #D_arry.append(to_there.loc) #將當下終點加入無人機的路徑矩陣
                        rx,ry = to_there.two_point_distance(from_here, "D")
                        #rx,ry= to_there.astar(from_here,"D")#呼叫astar 回傳astar路徑座標
                        #print('============D end===============')
                        #print(f'from_here:{from_here.loc} to_there :{to_there.loc} ')
                        #print(f'rx = {rx}')
                        #print(f'ry = {ry}')
                        #print('============D end===============')
                        fin_time = congestion_cal(rx,ry,V_R,"D") #計算f1效率
                        d_total_length += fin_time #計算該載具總效率
                        from_here = copy.deepcopy(to_there)#將當下的終點設為下次的起點 以改值不改址的方式
                        self.D_path_array = path_input(self.D_path_array,rx,ry,'D')#呼叫計算安全指數範圍函式
                        
                else:    
                    #D_arry.append(to_there.loc) #將當下終點加入無人機的路徑矩陣 
                    #rx,ry= to_there.astar(from_here,"D")#呼叫astar 回傳astar路徑座標
                    rx,ry = to_there.two_point_distance(from_here, "D")
                    
                    #print('============D===============')
                    #print(f'from_here:{from_here.loc} to_there :{to_there.loc} ')
                    #print(f'rx = {rx}')
                    #print(f'ry = {ry}')
                    #print('============D===============')
                    fin_time = congestion_cal(rx,ry,V_R,"D") #計算f1效率
                    d_total_length += fin_time #計算該載具總效率
                    from_here = copy.deepcopy(to_there)#將當下的終點設為下次的起點 以改值不改址的方式
                    self.D_path_array = path_input(self.D_path_array,rx,ry,'D')#呼叫計算安全指數範圍函式
                    
                    if(len(path_copy) == 0):
                       end = copy.deepcopy(to_there)#將當下的終點設為下次的起點 以改值不改址的方式
                       end.loc = (76,41) #起始點位置更改
                       #D_arry.append(to_there.loc) #將當下終點加入無人機的路徑矩陣
                       #rx,ry= end.astar(from_here,"D")#呼叫astar 回傳astar路徑座標
                       rx,ry = end.two_point_distance(from_here, "D")
                       #print('============D fin end ===============')
                       #print(f'from_here:{from_here.loc} to_there :{end.loc} ')
                       #print(f'rx = {rx}')
                       #print(f'ry = {ry}')
                       #print('============D fin end===============')
                       fin_time = congestion_cal(rx,ry,V_R,"D") #計算f1效率
                       d_total_length += fin_time #計算該載具總效率
                       from_here = copy.deepcopy(to_there)#將當下的終點設為下次的起點 以改值不改址的方式
                       self.D_path_array = path_input(self.D_path_array,rx,ry,'D')#呼叫計算安全指數範圍函式 
            #print(f"one loop check {end_check}")

        #print(f'V {V_arry}')
        #print(f'SDV {SDV_arry}')
        #print(f'D {D_arry}')
        total_length = v_total_length + sdv_total_length + d_total_length #計算該基因總效率
        total_array = self.V_path_array + self.SDV_path_array + self.D_path_array #三種載具的安全指數矩陣相加
        del(end)
        del(to_there)
        del(from_here)
        gc.collect()
        return np.round(total_length,2),total_array
         
    def _set_resilience(self):  #f2 計算應變能力

        Ward_D_V_sum_x = 0
        Ward_D_V_sum_y = 0
        Ward_SDV_V_sum_x = 0
        Ward_SDV_V_sum_y = 0
        loc_count = 0
        SDV_V_count = 0
        V_D_count = 0
        WardAVG_D_V_x = 0
        WardAVG_D_V_y = 0
        WardAVG_SDV_V_x = 0
        
        WardAVG_SDV_V_y = 0
        SDV_V_Location = []
        D_V_Location = []
        Ward_SDV_V = 0
        Ward_D_V = 0
        path_copy = self.path[:]
        path_names = [loc.name for loc in path_copy]
        path_loc = [loc.loc for loc in path_copy]

        
        for i in range(len(path_names)):    #算出沃德法所需的變數  已加入遇0省略 無須再count上加減
            if path_names[i] == "0":        #無人警車-警車全路徑點的平均
                loc_count +=1               #無人機-警車全路徑點的平均
            if(loc_count <= 2):      #如count為0表示目前為警車 如更改警車數量此處需更改       
                Ward_D_V_sum_x += path_loc[i][0]   #無人機對警車的全路徑點相加
                Ward_D_V_sum_y += path_loc[i][1] 
                Ward_SDV_V_sum_x += path_loc[i][0] #無人警車對警車的全路徑點相加
                Ward_SDV_V_sum_y += path_loc[i][1]
                SDV_V_count +=1
                V_D_count +=1
                #SDV_V_Location.append(Location(path_names[i],xs[i],ys[i])) #將無人警車對警車的全路徑點加入 SDV_V_Location
                #D_V_Location.append(Location(path_names[i],xs[i],ys[i]))   #無人機對警車的全路徑點加入 D_V_Location
                SDV_V_Location.append(Location(path_names[i],path_loc[i][0],path_loc[i][1])) #將無人警車對警車的全路徑點加入 SDV_V_Location
                D_V_Location.append(Location(path_names[i],path_loc[i][0],path_loc[i][1]))   #無人機對警車的全路徑點加入 D_V_Location
            elif(loc_count <= 4 and loc_count > 2):  #如count為1表示目前為無人警車 如更改無人警車數量此處需更改     
                Ward_SDV_V_sum_x += path_loc[i][0] #無人警車對警車的全路徑點相加
                Ward_SDV_V_sum_y += path_loc[i][1]
                if(path_names[i] != "0"): #因F2計算各點的相聚性，故起點不算
                    SDV_V_count +=1
                    SDV_V_Location.append(Location(path_names[i],path_loc[i][0],path_loc[i][1])) #將無人警車對警車的全路徑點加入 SDV_V_Location
            elif(loc_count > 4): #如count大於1表示目前為無人機 如更改無人機數量此處需更改
                Ward_D_V_sum_x += path_loc[i][0] #無人機對警車的全路徑點相加
                Ward_D_V_sum_y += path_loc[i][1]
                if(path_names[i] != "0"): #因F2計算各點的相聚性，故起點不算
                    V_D_count +=1
                    D_V_Location.append(Location(path_names[i],path_loc[i][0],path_loc[i][1])) #無人機對警車的全路徑點加入 D_V_Location
        
        WardAVG_D_V_x = np.round(Ward_D_V_sum_x / (V_D_count),2) #無人機對警車的全路徑點平均
        WardAVG_D_V_y = np.round(Ward_D_V_sum_y / (V_D_count),2) 
        WardAVG_SDV_V_x = np.round(Ward_SDV_V_sum_x / (SDV_V_count),2) #無人機對警車的全路徑點平均
        WardAVG_SDV_V_y = np.round(Ward_SDV_V_sum_y / (SDV_V_count),2)
   
        for i in D_V_Location: #沃德法算法 路徑的各點減去全路徑點的平均直的兩次方再開根號
            Ward_D_V += math.sqrt(pow((WardAVG_D_V_x - i.loc[0]),2) + pow((WardAVG_D_V_y - i.loc[1]),2))
        for i in SDV_V_Location:
            Ward_SDV_V += math.sqrt(pow((WardAVG_SDV_V_x - i.loc[0]),2) + pow((WardAVG_SDV_V_y - i.loc[1]),2))

        ward_ans = np.round(Ward_D_V,2) + np.round(Ward_SDV_V,2) #無人機對警車跟無人警車對警車的部分相加
        
        del Ward_D_V_sum_x
        del Ward_D_V_sum_y
        del Ward_SDV_V_sum_x
        del Ward_SDV_V_sum_y
        del loc_count
        del SDV_V_count
        del V_D_count
        del WardAVG_D_V_x
        del WardAVG_D_V_y
        del WardAVG_SDV_V_x        
        del WardAVG_SDV_V_y
        del SDV_V_Location
        del D_V_Location
        del Ward_SDV_V
        del Ward_D_V
        gc.collect()
        
        return np.round(ward_ans,2)
    
    def _set_safe_value(self,tol_array): #F3  掃描一次tol_array 看其中數值符合哪項 公式 = 安全矩陣 * 1 * 該項權重
        v_w = 0.5     #警車權重
        sdv_w = 0.3   #無人警車權重
        d_w = 0.2     #無人機
        tol_safe_value = 0
        x = 1
        for i in range (len(self.tol_array)): #掃描三種載具的安全指數矩陣相加，根據其中數值判斷為何種組合
            for j in range (len(self.tol_array[i])): 
                if int(self.tol_array[i][j]) == 1:                      #如果標記為1表示該點為警車
                    tol_safe_value += x * v_w * V_Safe[i][j]
                elif int(self.tol_array[i][j]) == 2:                    #如果標記為2表示該點為無人警車
                    tol_safe_value += x * sdv_w * V_Safe[i][j]
                elif int(self.tol_array[i][j]) == 4:                    #如果標記為4表示該點為無人機
                    tol_safe_value += x * d_w * V_Safe[i][j]
                elif int(self.tol_array[i][j]) == 3:                    #如果標記為3表示該點為警車 + 無人警車
                    tol_safe_value += x * (v_w + sdv_w) * V_Safe[i][j]
                elif int(self.tol_array[i][j]) == 5:                    #如果標記為5表示該點為警車 + 無人機
                    tol_safe_value += x * (v_w + d_w) * V_Safe[i][j]
                elif int(self.tol_array[i][j]) == 6:                    #如果標記為6表示該點為無人警車 + 無人機
                    tol_safe_value += x * (sdv_w + d_w) * V_Safe[i][j]
                elif int(self.tol_array[i][j]) == 7:                    #如果標記為7表示該點為無人警車 + 無人機 + 警車
                    tol_safe_value += x * (v_w + sdv_w + d_w) * V_Safe[i][j]
        

        return np.round(tol_safe_value,1)

class GeneticAlgo:
    def __init__(self, locs, level=10, populations=100, variant=3, mutate_percent=0.1, elite_save_percent=0.1,crossover_rate = 0.5):
        self.locs = locs
        #locs 要走的城市有哪些（注意locs是一個list裡面裝著Location物件）
        self.level = level
        #level 子代的代數（進化次數）
        self.variant = variant
        #variant為父代交配的基因數
        self.populations = populations
        #population 指的就是由幾個不同路徑組成一個群體
        self.mutates = int(populations * mutate_percent)       
        #mutate_percent 是說一個群體當中有多少比例是突變來的
        self.crossover = int(populations * crossover_rate)
        self.crossover_rate = crossover_rate
        self.elite = int(populations * elite_save_percent)
        #elite_save_percent 是說在一群路徑當中，最短的幾％路徑被視為精英路徑（這邊是0.1=10%）
    # 建立第一代的路徑群體    
    
    def _find_path(self):
        #負責從self.locs(要走的城市)當中隨機生成可行的路徑（每個都要走到），然後回傳這個路徑回來
        #0為一條載具路徑的開始(除最開始的起點)
        # locs is a list containing all the Location obj
        locs_copy = self.locs[:]
        count = 0
        path = []
        to_there = Location('start',0,0)

        while locs_copy: #根據隨機選擇依序選擇一座標加入路徑中
            temp = rd.choice(locs_copy)   
            if(count % 3 == 0 and count != 0): #每經過2點加入一個0 此處如更改載具數量此處需更改
                to_there = Location('0',0,0)  #這邊之後要根據機種更改起始點
                path.append(to_there)
            to_there = locs_copy.pop(locs_copy.index(temp))
            path.append(to_there)
            count +=1
            
        return path
    
    def zero_change(self,index,end): #根據路徑0的位置進行判斷，判斷方法為當下0的位置與下個0的位置是否差距n個位置，n的公式為(巡邏點總數 / 車數)
    #index 為標記該路徑0的位置
        for i in range(len(index) - 1):
            if index[i + 1] - index[i] < 4:
                # 2 為要設置的點 限制  (巡邏點總數 / 車數)
                #print(f'index[i + 1] = {index[i + 1]} - index[i] = {index[i]}')
                check =  True
                break
            elif end  - index[i] < 4 :
                check =  True
                break
            else:
                check =  False
        return i,check
    
    def _init_routes(self):
    #那這個function就會負責呼叫_find_path來取得路徑，再用這個路徑來生成Route物件。生成的次數就取 決於self.populations(群體的數量)是多少囉。
        routes = []
        for _ in range(self.populations):
            path = self._find_path()
            routes.append(Route(path))
        return routes
    
       
    def _get_next_route(self, routes):    
        bin_selec = []
        
        start2 = time.time()
        pqc,q = fitness(routes) #計算gpsiff
        for i in range(len(routes)):
            routes[i].gpsiff = pqc[i]
            routes[i].q = q[i]
        routes.sort(key=lambda x: x.gpsiff, reverse=True) #這個函式會把一個群體（路徑們）按照他們的gpsiff大小排序 由大排至小
        #for i in routes:
            #path_names = [loc.name for loc in i.path]
            #print(f'path : {path_names}')
            #print([(loc.loc[0], loc.loc[1]) for loc in i.path])
            #print(f'safe_value: {i.safe_value} length: {i.length} resilience: {i.resilience} gpsiff: {i.gpsiff}')
        #print("===============================get_next_route=============================")
        
############################################################################           
        for _ in range(self.populations):
            com_1, com_2 = rd.choices(routes[:], k=2)  #我們從菁英路徑們當中的前四名隨機取出兩個作為父和母。          
            if com_1.gpsiff >= com_2.gpsiff:
                bin_selec.append(com_1)
            else:
                bin_selec.append(com_2)
############################################################################
        end2 = time.time()
        print(f'二元競爭  耗費時間 {np.round(end2 - start2)}')    
        crossovers = self._crossover(bin_selec) #接著呼叫_crossover來取得繁衍後的子代。
        start1 = time.time()
        temp = crossovers[:] + routes # 要在確認要用2元競爭下的父代還是原父代
        pqc,q = fitness(temp) #計算gpsiff
        
        for i in range(len(routes)):           
            temp[i].gpsiff = pqc[i]
            temp[i].q = q[i]
        next_generation = []
        another = []
        '''
        path = 'output.txt'
        f = open(path, 'a')
        for i in temp:
            bin_selec_name = [loc.name for loc in i.path]
            f.write(str(bin_selec_name) + "\n")
            f.write("safe_value =  "+str(i.safe_value) +"length =  "+str(i.length) +"resilience = " 
                    +str(i.resilience) + " pqc = "+ str(i.gpsiff) +" q = "+ str(i.q) +"\n")
             
            #print(f'elite {bin_selec_name} pqc :{i.gpsiff}')           
            #print(f'pqc :{i.gpsiff} safe_value: {i.safe_value} efficiency: {i.length} resilience: {i.resilience}')
        
        f.write("========================================================================\n")
        f.close()
        '''
        pop_count = 0
        for i in temp:
            if ((i.q == 0) and (pop_count < self.populations)):
                next_generation.append(i)
                pop_count += 1
            else:
                another.append(i) 
        
        #print(f'pop_count = {pop_count}')
        if pop_count < self.populations:
            another.sort(key=lambda x: x.gpsiff, reverse=True)
            another_count = 0
            
            while pop_count < self.populations:
                next_generation.append(another[another_count])
                pop_count += 1
                another_count += 1
                #print(f'pass2_pop_count = {pop_count}')
        end1 = time.time()
        print(f'μ+λ 選擇  耗費時間 {np.round(end1 - start1)}')                    
        """    
        for _ in range(self.populations):
            com_1, com_2 = rd.choices(temp[:], k=2)  #我們從菁英路徑們當中的前四名隨機取出兩個作為父和母。          
            if com_1.gpsiff >= com_2.gpsiff:
                next_generation.append(com_1)
            else:
                next_generation.append(com_2)
        """        
        #next_generation = crossovers[:] + elites #下一代為交配後的子代 + 菁英群體
        
        #path = 'output.txt'
        #f = open(path, 'a')
        #f.write("=================交配後的子代 + 菁英群體=================\n")
        '''
        print("=========================交配後的子代 + 菁英群體=================================") 
        for i in next_generation:
            i.path = path_fix(i.path)
            next_generation_path_names = [loc.name for loc in i.path]
            
            
            print(f'next_generation {next_generation_path_names}')  
            print(f'safe_value: {i.safe_value} length: {i.length} resilience: {i.resilience} gpsiff: {i.gpsiff} q:{i.q}')
            #f.write(str(next_generation_path_names) + "\n")
            #f.write("safe_value =  "+str(i.safe_value) +"length =  "+str(i.length) +"resilience = " +str(i.resilience)
                   #+ " pqc = "+ str(i.gpsiff) +" q = "+ str(i.q)  + "\n")
        '''          
        print(f'下一代路徑數 = {len(next_generation)}')
        #f.write("=======================================================\n")
        #f.write("下一代路徑數 = " + str(len(next_generation)) + "\n")
        #f.write("=======================================================\n")
        #f.close()
        #print("=====================================================")
        return next_generation
        #最後生成新的子代。
        
       
    def _crossover(self, elites): #交配
        # Route is a class type
        start = time.time()
        normal_breeds = []
        mutate_ones = []
        normal_ones = []
        for _ in range(self.populations - self.mutates):
            father, mother = rd.choices(elites[:8], k=2)  #我們從菁英路徑們當中的前四名隨機取出兩個作為父和母。          
            index_start = rd.randrange(0, len(father.path) - self.variant - 1)
            #print(f'index_start {index_start}') 上次改這邊
            father_gene = father.path[index_start: index_start + self.variant] #由菁英路徑由指定切割位置取出父代的基因            
            father_gene_names = [loc.name for loc in father_gene]
            #for i in father_gene:
                #if i.name in "1" or i.name in "0":
                    #print("ex")
                #print(i.name)
            #print('==')

            #我們從菁英路徑們當中的前四名隨機取出兩個作為父和母。
       
            #========不處理 0 只處理父代其他相同的東西========
            #移除母親染色體當中含有父親那小節染色體的部分，同時決定要從哪裡放入父親那小節染色體。
            mother_gene = []
            for gene in mother.path :
                if gene.name != "0" :
                    if bool(gene.name in father_gene_names ) == False:
                        mother_gene.append(gene) #去除母代中與父代相同的基因
                else:
                    mother_gene.append(gene)

            #mother_gene_cut = rd.randrange(1, len(mother_gene)) #選擇母代切割的位置

            #print(mother_gene_cut)
            # create new route path
            next_route_path = mother_gene[:index_start] + father_gene + mother_gene[index_start:] #下代基因為 母代切割 + 父代 + 母代切割
            next_route_names = [loc.name for loc in next_route_path] 

     
            #===============修補機制================================
            # 刪除多餘的0 / 刪除開頭為0 /刪除結尾為0
            index = [i for i,v in enumerate(next_route_names) if v == "0"]
            
            while (index[0] == 0) or (index[len(index) - 1] == len(next_route_path) - 1) or next_route_names.count("0") > 9:
                #while判斷 第一個為開頭為0 第二個為最後為0 第三個為0超過需求數(需求數如果有更改車輛數量或點數量需進行更改) 
                
                if index[0] == 0 :                                       #如果路徑開頭為0 則刪除第一個路徑座標
                      del next_route_path[index[0]]               
                elif index[len(index) - 1] == len(next_route_path) - 1 : #如果路徑最後為0 則刪除最後一個路徑座標
                      del next_route_path[index[len(index) - 1]]
                else:                                                    #如果為超過需求數則隨機選擇一0的座標進行刪除                     
                    rand = rd.choice(index)
                    del next_route_path[rand]
                    #print("test 3")           
                next_route_names = [loc.name for loc in next_route_path]
                index = [i for i,v in enumerate(next_route_names) if v == "0"] #再算一次0的位置

            # 檢查剩下的path 是否需要補0
            while len(index) < 9:                                       #如果0的數量小於需求數 則進行修補(需求數如果有更改車輛數量或點數量需進行更改) 
                rand = rd.randrange(1, len(next_route_names) - 1)       #隨機選一位置進行插入 以撇除頭與尾
                next_route_path.insert(rand,Location('0',0,0))
                
                next_route_names = [loc.name for loc in next_route_path]
                index = [i for i,v in enumerate(next_route_names) if v == "0"]
                #print("add 0")   改這
            
            #start_2 = time.time()
            i,check = self.zero_change(index,len(next_route_names)) #計算0的位置是否符合規則
               
            while check == True or index[0] <= 3: #index[0] <= n n為需求數 (需求數如果有更改車輛數量或點數量需進行更改)                    
                rand = rd.randrange(1, len(next_route_names) - 1)#隨機選擇一位置
                next_route_path[index[i]],next_route_path[rand] = next_route_path[rand] , next_route_path[index[i]] #將違規的0位置與選到的位置替換
                next_route_names = [loc.name for loc in next_route_path]
                index = [i for i,v in enumerate(next_route_names) if v == "0"]

                i,check = self.zero_change(index,len(next_route_names)) #再進行一次計算0的位置是否符合規則，如不符合 則替換到符合規則為止
                #目前以暴力的方式進行替換 如日後有更好的方式需進行更改節省時間    
                #breakpoint()
            #end_2 = time.time() 
            #print(f"一次修補後半耗費時間{np.round(end_2 - start_2,2)}")
            #============================================================    
            #接著next_route_path就是子代路徑
            next_route = Route(next_route_path)
            #然後我們用子代路徑建立Route物件。
            #==========================交配機率==============================
            #rate = 0.6
            # add Route obj to normal_breeds
            normal_ones.append(next_route)
            normal_breeds = rd.choices(normal_ones, k=self.crossover) 
            '''if (self.crossover_rate > round(rd.uniform(0.1,0.9),1)):
                normal_breeds.append(next_route)'''
            #===============================================================
            #===============================================================================================================
            # for mutate purpose
            copy_father = copy.deepcopy(father)
            idx = range(len(copy_father.path))
            gene1, gene2 = rd.sample(idx, 2)
            #print(f'gene1 = {copy_father.path[gene1].name} gene2 = {father.path[gene2].name}')
            while(copy_father.path[gene1].name == '0' or copy_father.path[gene2].name == '0'):
                gene1, gene2 = rd.sample(idx, 2)
            #兩座城市突變 現方法遇0重新rand直到沒0 之後兩城市位置互換
            
            copy_father.path[gene1], copy_father.path[gene2] = copy_father.path[gene2], copy_father.path[gene1]
          
            #在第五行中我們隨機調換父路徑當中的兩座城市作為突變，加到mutate_ones裡頭。
            mutate_ones.append(copy_father)
        #重要 class只會在創建第一次時回傳函數 所以在繁殖突變是要在呼叫一次目標函是    
        #print("============================繁殖================================")  
        end = time.time()
        print(f"修補耗費時間{np.round(end - start,2)}")
        start = time.time()
        for i in range(len(normal_breeds)):
            normal_breeds[i].path = path_fix(normal_breeds[i].path) #方便管理在每次交配完的路徑都進行起點歸零化
            normal_breeds[i].V_path_array = create_path_array(x,y) #創建警車的安全指數矩陣 供f3使用
            normal_breeds[i].SDV_path_array = create_path_array(x,y)#創建無人警車的安全指數矩陣 供f3使用
            normal_breeds[i].D_path_array = create_path_array(x,y)#創建無人機的安全指數矩陣 供f3使用
                
            normal_breeds_names = [loc.name for loc in normal_breeds[i].path]
            normal_breeds[i].safe_value,normal_breeds[i].tol_array = normal_breeds[i]._set_length()
            normal_breeds[i].resilience = normal_breeds[i]._set_resilience() #計算 f2 應變能力 (回傳應變能力)
            normal_breeds[i].safe_value = normal_breeds[i]._set_safe_value(normal_breeds[i].tol_array) #計算 f3 安全指數 (回傳安全指數
            #print(normal_breeds_names)
            #print(f'safe_value: {normal_breeds[i].safe_value} efficiency: {normal_breeds[i].length} resilience: {normal_breeds[i].resilience}')
        end = time.time()
        print("=========================")
        print(f'繁殖耗費時間 {np.round(end - start,2)}')
        #print("====================================================================")
        start = time.time()
        mutate_breeds = rd.choices(mutate_ones, k=self.mutates)   
        #最後一行是因為我們最後只要self.mutates這麼多個突變個數
        #muate_percent 控制突變後路徑的數量
        #print("========================突變================================")
        for i in range(len(mutate_breeds)):
            mutate_breeds[i].path = path_fix(mutate_breeds[i].path) #方便管理在每次交配完的路徑都進行起點歸零化
            mutate_breeds[i].V_path_array = create_path_array(x,y) #創建警車的安全指數矩陣 供f3使用
            mutate_breeds[i].SDV_path_array = create_path_array(x,y)#創建無人警車的安全指數矩陣 供f3使用
            mutate_breeds[i].D_path_array = create_path_array(x,y)#創建無人機的安全指數矩陣 供f3使用
            
            
            mutate_breeds_names = [loc.name for loc in mutate_breeds[i].path]
            mutate_breeds[i].safe_value,mutate_breeds[i].tol_array = mutate_breeds[i]._set_length()
            mutate_breeds[i].resilience = mutate_breeds[i]._set_resilience() #計算 f2 應變能力 (回傳應變能力)
            mutate_breeds[i].safe_value = mutate_breeds[i]._set_safe_value(mutate_breeds[i].tol_array) #計算 f3 安全指數 (回傳安全指數
            #print(mutate_breeds_names)
            #print(f'safe_value: {mutate_breeds[i].safe_value} efficiency: {mutate_breeds[i].length} resilience: {mutate_breeds[i].resilience}')
        #print("====================================================================")
        result = normal_breeds + mutate_breeds
        end = time.time()
        print(f'突變耗費時間 {np.round(end - start,2)}')
        print(f'突變+繁殖數量 {len(result)}')
        return result
        #===============================================================================================================    
    
    
                
    def evolution(self): #進化
        global now_level,now_length,now_safe_value,now_resilience
        routes = self._init_routes()

        
        for _ in range(self.level):
            start = time.time()
            routes = self._get_next_route(routes)
            end = time.time()
            print(f'一代耗費時間 {np.round(end - start,2)}')
            #print("==================================================")
            pqc,q = fitness(routes) #結合子母代染色體並計算gpsiff
            
            for i in range(len(routes)):
                routes[i].gpsiff = pqc[i]
                routes[i].q = q[i]
                
            print(f'level = {_}')
            if (_ + 1) % 1 == 0:
                routes.sort(key=lambda x: x.gpsiff, reverse=True) #使用gpsiff進行排序取出最佳的   
                '''
                print("=========================第三方篩選中=================================") 
                for i in routes:
                    elites_path_names = [loc.name for loc in i.path]

                    
                    print(f'evolution {elites_path_names}')           
                    print(f'pqc :{i.gpsiff} safe_value: {i.safe_value} efficiency: {i.length} resilience: {i.resilience} Q = {i.q}')
                print("=====================================================")
                '''
                length_Temp = []
                safe_value_Temp = []
                resilience_Temp = []
                path_Temp = []
        
                for i in range (len(routes)):                                        
                    if routes[i].q == 0:
                        third_pop.append(routes[i])
                        
                        
                        
                        length_Temp.append(routes[i].length)
                        safe_value_Temp.append(routes[i].safe_value) 
                        resilience_Temp.append(routes[i].resilience)
                        path_Temp.append(routes[i].path)
                
                        
                third_pop_cal(_)

                now_level.append(_ + 1)
                now_length.append(length_Temp)
                now_safe_value.append(safe_value_Temp)
                now_resilience.append(resilience_Temp)  
                now_path.append(path_Temp)
                '''
                print("第三方基因")
                print(now_length)
                print(now_safe_value)
                print(now_resilience)
                '''
            '''print("=========================third_pop=================================") 
            for i in third_pop:
                         
                print(f'pqc :{i.gpsiff} safe_value: {i.safe_value} efficiency: {i.length} resilience: {i.resilience} Q = {i.q}')
            print("=====================================================")'''
        #######################################################       
        pqc,q = fitness(routes) #結合子母代染色體並計算gpsiff
        
        for i in range(len(routes)):
            routes[i].gpsiff = pqc[i]
            routes[i].q = q[i]  
            
        routes.sort(key=lambda x: x.gpsiff, reverse=True) #使用gpsiff進行排序取出最佳的
        ########################################################
        print("=========================最終一代路徑=================================") 
        for i in routes:
            elites_path_names = [loc.name for loc in i.path]
            
            
            print(f'evolution {elites_path_names}')           
            print(f'pqc :{i.gpsiff} safe_value: {i.safe_value} efficiency: {i.length} resilience: {i.resilience}')
        print("=====================================================")
        print(f'路徑數 = {len(routes)}')
        print([(loc.name) for loc in routes[0].path])
        print([(loc.loc[0], loc.loc[1]) for loc in routes[0].path])
        return routes[0].path, routes[0].length, routes[0].safe_value, routes[0].resilience, routes[0].gpsiff
    #初始化群體 -> 繁衍新群體(重複self.level次（子代代數）) -> 得到最後的群體


if __name__ == '__main__':
    start = time.time()
    now_level = []
    now_length = []
    now_safe_value = []
    now_resilience = []
    now_path = []
    best_path = []
    
    
    my_locs, xs, ys, cities = create_locations()
    my_algo = GeneticAlgo(my_locs, level= 10, populations=10, variant=4, mutate_percent=0.1, elite_save_percent=0.2,crossover_rate = 0.6)
    best_route, best_route_length, best_route_safe_value,best_route_resilience,best_route_pqc = my_algo.evolution()
    #best_route.append(best_route[0]) 回原點
    print([loc.name for loc in best_route], best_route_length,best_route_safe_value,best_route_resilience,best_route_pqc)
    print([(loc.loc[0], loc.loc[1]) for loc in best_route])
    #print([(loc.name) for loc in best_route])
    best_route_name = [loc.name for loc in best_route]
    best_route_loc = [(loc.loc[0], loc.loc[1]) for loc in best_route]
    plot_show(best_route)
    #best_F1,best_F2,best_F3 = third_party_cal(now_level, now_length, now_safe_value, now_resilience,now_path)
    best_F1,best_F2,best_F3 = third_pop_lim(all_best_f1,all_best_f2,all_best_f3)
    
    best_path = [best_route_length,best_route_resilience,best_route_safe_value,best_route_loc,best_route_name]
    print(f'best_F1 f1 = {best_F1.length} f2 = {best_F1.resilience} f3 = {best_F1.safe_value}')
    print(f'best_F2 f1 = {best_F2.length} f2 = {best_F2.resilience} f3 = {best_F2.safe_value}')
    print(f'best_F3 f1 = {best_F3.length} f2 = {best_F3.resilience} f3 = {best_F3.safe_value}')
    export_var("best_file",best_F1,best_F2,best_F3,best_path)
    export_var("all_best_f1",all_best_f1)
    export_var("all_best_f2",all_best_f2)
    export_var("all_best_f3",all_best_f3)
    
    third_pop.sort(key=lambda x: x.gpsiff, reverse=True)
    print("=========================endthird_pop=================================") 
    for i in third_pop:
                 
        print(f'pqc :{i.gpsiff} safe_value: {i.safe_value} efficiency: {i.length} resilience: {i.resilience} Q = {i.q}')
    print("=====================================================")

    
    
    end = time.time()
    print(f'time = {round(end - start,2)} s')
    #print(all_best_f1)
    #print(all_best_f2)
    #print(all_best_f3)
    