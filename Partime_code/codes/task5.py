import sys
path = "/home/s1810235/part_time/PtNafion/"
sys.path.insert(0, path)
from plot import *
from opt_GMM_2 import *
import numpy as np
import os 
from itertools import combinations
import re
import time

os.makedirs(path + "image/task5/joint_org",exist_ok=True )
os.makedirs(path + "image/task5/joint_diff",exist_ok=True )
def load_file_path(path):
    list_file_path = []
    count = 0
    for i in range(len(path)):
        for feature in os.listdir(path[i]):
            if not feature.endswith(".DS_Store"):
                for file_to_append in os.listdir(str(path[i])+ str(feature) + "/"):
                    if(file_to_append.endswith(".txt") and not (file_to_append.startswith("._"))):
                        list_file_path.append(path[i] + feature + "/"+ file_to_append)
    return list_file_path                   

def load_file_section3(p1,t1,v,p2):
    sub_file_path1 = path+p1
    sub_file_path2 = path+p2
    list_feature1 = []
    list_feature2 = []
    for feature in os.listdir(sub_file_path1):
        if not feature.endswith(".DS_Store"):
            for file in os.listdir(sub_file_path1 + str(feature)+ "/"):
                # print(feature)  
                Pt1 = (np.loadtxt(sub_path+ str(feature)+"/" + t1 + v1 + "_" + str(feature)+".txt"))
                list_feature1.append(Pt1)
    for feature in os.listdir(sub_file_path2):
        if not feature.endswith(".DS_Store"):
            for file in os.listdir(sub_file_path2 + str(feature)+ "/"):
                # print(feature)  
                Pt1 = (np.loadtxt(sub_path+ str(feature)+"/" + t2 + v2 + "_" + str(feature)+".txt"))
                list_feature1.append(Pt1)

def combination(list_file_path):
    combination_list = []
    comb = combinations([i for i in range(len(list_file_path))],2)
    for i in comb:
        tmp = []
        for j in i:
            tmp.append(list_file_path[j])
        combination_list.append(tmp)
    return combination_list

    
def get_file_name(string):
    return (str(re.findall(r'PtNafion/(.*?).txt', string))).replace("/","_")


def plot_disstributions(list_combination):
    count = 0
    print(len(list_combination))
    for i in list_combination:
        count += 1
        tmp  = []
        name = []
        

        for j in range(len(i)):
            tmp.append(np.ravel((np.loadtxt(i[j]))))
            name.append(get_file_name(i[j]))
        min_x = np.min((tmp[0])) ######x tmp[0],y=temp[1]
        max_x = np.max((tmp[0]))
        min_y = np.min((tmp[1])) ######x tmp[0],y=temp[1]
        max_y = np.max((tmp[1]))
        
        print("_"*50)
        print("{}____{}____".format(min,max))
        print("processing.........")   
        start = time.time()
        joint_plot_fill(tmp[0], tmp[1], str(name[0]), str(name[1]), path + "image/task5/joint_org/{}___{}.pdf".format(str(name[0]),str(name[1])) , min_x ,max_x,min_y,max_y )
        print("finish............" + "Time take:{}".format(time.time()-start)+"to run"+"file num:{}".format(count)) 
def remove_nan(matrix1,matrix2):
    nan_of_matrix1 = np.argwhere(np.isnan(matrix1))
    nan_of_matrix2 = np.argwhere(np.isnan(matrix2))
    
    matrix1 = np.delete(matrix1,np.concatenate((nan_of_matrix2,nan_of_matrix2),axis=0))
    matrix2 = np.delete(matrix2,np.concatenate((nan_of_matrix2,nan_of_matrix2),axis=0))
    return matrix1,matrix2

def plot_nan_disstributions(list_combination):
    count = 0
    print(len(list_combination))
    list_nan_combination = []
    for i in list_combination:
        if(np.isnan(np.ravel((np.loadtxt(i[0])))).any()==True or np.isnan(np.ravel((np.loadtxt(i[1])))).any()==True):
            print(i)
            list_nan_combination.append(i)
    # print(len(list_nan_combination))
    for i in list_nan_combination:
        count += 1
        tmp  = []
        name = []
        for j in range(len(i)):
            tmp.append(np.ravel((np.loadtxt(i[j]))))
            name.append(get_file_name(i[j]))
        tmp[0],tmp[1] = remove_nan(tmp[0],tmp[1])
        
        min_x = np.min((tmp[0])) ######x tmp[0],y=temp[1]
        max_x = np.max((tmp[0]))
        min_y = np.min((tmp[1])) ######x tmp[0],y=temp[1]
        max_y = np.max((tmp[1]))
        
        print("_"*50)
        print("{}____{}____".format(min,max))
        print("processing.........")   
        start = time.time()
        joint_plot_fill(tmp[0], tmp[1], str(name[0]), str(name[1]), path + "image/task5/joint_org/{}___{}.pdf".format(str(name[0]),str(name[1])) , min_x ,max_x,min_y,max_y )
        print("finish............" + "Time take:{}".format(time.time()-start)+"to run"+"file num:{}".format(count)) 
        
        
                
def main():
    a = load_file_path([path + "CCM-Nafion/",path + "CCMcenter/"])

    b = combination(a)
    plot_nan_disstributions(b)
    

if __name__ == "__main__":main()