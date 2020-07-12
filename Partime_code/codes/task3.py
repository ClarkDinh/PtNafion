import sys
# path = "/home/s1810235/part_time/PtNafion/"
path = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion/input/2020-06-17-NewRequest/txt/"

sys.path.insert(0, path)
from plot import plot_density
import os
import numpy as np
import glob
from copy import copy, deepcopy
os.makedirs(path+ "image/task3",exist_ok=True)
os.makedirs(path+ "feature/task3",exist_ok=True)

def convert_bulk_bkg(output_feature,output_image,p,sub_square_size):
    sub_path = path + p +"/" 
    list_matrix = []
    for feature in os.listdir(sub_path):
        if feature == "morphology" :
            if not feature.endswith(".DS_Store"):
                
                for file_init in os.listdir(sub_path + str(feature)+ "/"):
                    if file_init.endswith(".txt"):
                        if not file_init.startswith("._"):
                            matrix = (np.loadtxt(sub_path + str(feature)+ "/"+file_init))
                            if(np.max(matrix)< 4):
                                matrix = np.where(matrix > 0 , "bulk","bkg" )
                                matrix,matrix_transform_to_plot = morph(matrix,sub_square_size)
                                np.savetxt(path + output_feature + "bulk_label_" + p +str(file_init) ,matrix,fmt="%s",delimiter="  ")
                                np.savetxt(path + output_feature + "num_bulk_label"+ p + str(file_init),matrix_transform_to_plot,fmt="%s",delimiter="  ")
                                matrix_transform_to_plot = matrix_transform_to_plot.astype(np.float)
                                plot_density(matrix_transform_to_plot, path + output_image + "bulk_label_"+p + str(file_init.replace(".txt","")),str(file_init.replace(".txt","")) ,  cmap_name="bwr", vmin=None, vmax=None)
                            elif(np.max(matrix)>4):
                                matrix = np.where(matrix > 4 , "bulk","bkg" )
                                matrix,matrix_transform_to_plot = morph(matrix,sub_square_size)
                                np.savetxt(path + output_feature + "bulk_label_" + p  +str(file_init) ,matrix,fmt="%s",delimiter="  ")
                                np.savetxt(path + output_feature + "num_bulk_label"+ p + str(file_init),matrix_transform_to_plot,fmt="%s",delimiter="  ")
                                matrix_transform_to_plot = matrix_transform_to_plot.astype(np.float)
                                plot_density(matrix_transform_to_plot, path + output_image + "bulk_label_"+p + str(file_init.replace(".txt","")),str(file_init.replace(".txt","")) ,  cmap_name="bwr", vmin=None, vmax=None)
                        
                                      

def transform_to_plot(matrix):
    
    
    matrix = np.where(matrix == "bulk", 1 , matrix)
    matrix = np.where(matrix == "bkg", -1 , matrix)
    matrix = np.where(matrix == "surf", 0 , matrix)
    
    return matrix
def padding(matrix):
    return np.pad(matrix,((2,2),(2,2)),mode='constant',constant_values= "-2")

def check_feature(matrix,matrix_duplicate,x,y,sub_square_size):
    #window size 5x5 and both
    # if(sub_square_size == 35 and ( matrix[x-2,y-2] == "bkg" or matrix[x-2,y-1] == "bkg" or matrix[x-2,y] == "bkg" or matrix[x-2,y+1] == "bkg" or matrix[x-2,y+2] == "bkg" or matrix[x+2,y] == "bkg" or matrix[x-1,y-2] == "bkg" or matrix[x-1,y-1] == "bkg" or matrix[x-1,y] == "bkg" or matrix[x-1,y+1] == "bkg" or matrix[x-1,y+2] == "bkg" or matrix[x,y-2] == "bkg" or matrix[x,y-1] == "bkg" or matrix[x,y+1] == "bkg" or matrix[x,y+2] == "bkg" or matrix[x+1,y-2] == "bkg" or matrix[x+1,y-1] == "bkg" or matrix[x+1,y] == "bkg" or matrix[x+1,y+1] == "bkg" or matrix[x+1,y+2] == "bkg" or matrix[x+2,y-2] == "bkg" or matrix[x+2,y-1] == "bkg" or matrix[x+2,y] == "bkg" or matrix[x+2,y+1] == "bkg" or matrix[x+2,y+2] == "bkg")):
    #     print("FOUND SURF________________________________________________")
    #     matrix_duplicate[x-2,y-2] == "surf"
    # else :
    #     matrix_duplicate[x-2,y-2]
    
    # if(sub_square_size == 5 and ( matrix[x-2,y-2] == "bkg" or matrix[x-2,y-1] == "bkg" or matrix[x-2,y] == "bkg" or matrix[x-2,y+1] == "bkg" or matrix[x-2,y+2] == "bkg" or matrix[x-1,y-2] == "bkg" or matrix[x-1,y+2] == "bkg" or matrix[x,y-2] == "bkg" or matrix[x,y+2] == "bkg" or matrix[x+1,y-2] == "bkg"    or matrix[x+1,y+2] == "bkg" or matrix[x+2,y-2] == "bkg" or matrix[x+2,y-1] == "bkg" or matrix[x+2,y] == "bkg" or matrix[x+2,y+1] == "bkg" or matrix[x+2,y+2] == "bkg")):
    #     # print("FOUND SURF________________________________________________")
    #     matrix_duplicate[x-2,y-2] == "surf"
    # else :
    #     matrix_duplicate[x-2,y-2]

    if(matrix[x,y]=="bulk" and sub_square_size == 3 and (matrix[x-1,y-1] == "bkg" or matrix[x-1,y] == "bkg" or matrix[x-1,y+1] == "bkg" or matrix[x,y-1] == "bkg" or matrix[x,y+1] == "bkg" or matrix[x+1,y-1] == "bkg" or matrix[x+1,y] == "bkg" or matrix[x+1,y+1] == "bkg")):
        print("FOUND SURF________________________________________________")
        matrix_duplicate[x-2,y-2] = "surf"
        print(matrix_duplicate[x-2,y-2])
    else :
        matrix_duplicate[x-2,y-2]

    return matrix_duplicate[x-2,y-2]
    

def morph(elem,sub_square_size):
    matrix_duplicate = elem
    matrix_transform_to_plot = elem
    matrix = padding(elem)
    
    for row in range(len(matrix)):
        for cell in range(len(matrix)):
            if matrix[row,cell] == "-2":
                # print("Padding part")
                pass
            else:
                matrix_duplicate[row-2,cell-2] = check_feature(matrix,matrix_duplicate,row,cell,sub_square_size)
                
                
    print(matrix_duplicate)
    matrix_transform_to_plot = transform_to_plot(matrix_duplicate)
    print(matrix_transform_to_plot)


            
    return matrix_duplicate,matrix_transform_to_plot

def main():
    convert_bulk_bkg("feature/task3/","image/task3/","CCM-Nafion",3)
    convert_bulk_bkg("feature/task3/","image/task3/","CCMcenter",3)
    
    

    
    
        

if __name__ == "__main__":main()