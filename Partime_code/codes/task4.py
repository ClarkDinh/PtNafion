import sys
import torch
cuda = torch.device('cuda') 
path = "/home/s1810235/part_time/PtNafion/"
sys.path.insert(0, path)
from plot import plot_density
# from opt_GMM_2 import *
# from dmin import *
# from constant import *
import numpy as np
import os 
from itertools import combinations
import re
import math  
from scipy.spatial import distance
import time
from tqdm import tqdm
os.makedirs(path + "feature/task4/",exist_ok=True)
os.makedirs(path + "image/task4/")
def euclidean_broadcast(x,y,batch_size=10000,device='cuda'):
	# # Nguyen: x stands for bulk index, y stand for surface index
	# # return distance from all possible surface "y" to all possible "bulk" x
    tensor_y = torch.tensor(y,dtype=torch.float32,device=device)
    results=None
    for batch_idx in tqdm(range(len(x)//batch_size+1)):
        b_x = x[batch_idx*batch_size:(batch_idx+1)*batch_size]
        tensor_x = torch.tensor(b_x,dtype=torch.float32,device=device)
        dist = torch.sqrt(torch.sum((tensor_x[:,None]-tensor_y[None,:])**2,dim=-1)).cpu().detach().numpy()
        if results is None:
            results=dist
        else:
            results = np.append(results,dist,axis=0)
    return results
def get_file_name(string):
    return ("dmin"+str(re.findall(r'num_bulk_label(.*?).txt', string)))
def dmin_v3(path_,filename,feature_dir,image_dir):
    matrix_1_index = [] # # Nguyen: matrix_1_index index of bulk 
    matrix_0_index = [] # # Nguyen: matrix_0_index index of surface 
    matrix_distance = []
    matrix = np.loadtxt(path_ + filename)
    matrix = np.where(matrix==-1,np.nan,matrix)
    print(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] == 1:
                matrix_1_index.append([i,j])
            elif matrix[i,j] == 0:
                matrix_0_index.append([i,j])
    print("matrix_0 and 1 ")
    all_array_of_distance = euclidean_broadcast(matrix_1_index, matrix_0_index)
    matrix_distance = np.min(all_array_of_distance,axis=-1)
    for idx,i in enumerate(matrix_1_index):
        matrix[i[0],i[1]] = matrix_distance[idx]

    np.savetxt(feature_dir + get_file_name(filename)+".txt",matrix,delimiter=" ")
    #vì nan sẽ làm tính min max trong hàm plot bị lỗi, nên set nan thành 1 giá trị nào đó khác 0(surf) và âm thì nhìn thấy màu rõ hơn vì giá trị distance toàn là dương. 
    test[np.isnan(test)] = -30 
    plot_density(matrix, image_dir ,get_file_name(filename),  cmap_name="bwr", vmin=None, vmax=None)


def main():
    list_file_name = [ "num_bulk_labelCCM-NafionADT5k1V_morphology.txt","num_bulk_labelCCM-NafionADT5k04V_morphology.txt","num_bulk_labelCCM-NafionADT15k1V_morphology.txt","num_bulk_labelCCM-NafionADT15k04V_morphology.txt","num_bulk_labelCCM-NafionFresh1V_morphology.txt","num_bulk_labelCCM-NafionFresh04V_morphology.txt","num_bulk_labelCCMcenterADT5k1V_morphology.txt","num_bulk_labelCCMcenterADT5k04V_morphology.txt","num_bulk_labelCCMcenterADT15k1V_morphology.txt","num_bulk_labelCCMcenterADT15k04V_morphology.txt","num_bulk_labelCCMcenterFresh1V_morphology.txt","num_bulk_labelCCMcenterFresh04V_morphology.txt"] 
    for filename in list_file_name:
        dmin_v3(path + "feature/task3/",filename,path + "feature/task4/",path + "image/task4/")

if __name__ == "__main__":main()



                                

                             

                        





    

