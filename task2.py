import sys
path = "/home/s1810235/part_time/PtNafion/"
sys.path.insert(0, path)
from plot import plot_density
import numpy as np
import os 
os.makedirs(output_direction_feature + diff,exist_ok=True)
os.makedirs(output_direction_image + diff,exist_ok=True)

def feature(input_direction,output_direction_feature,output_direction_image,diff):
    for file_init in os.listdir(input_direction+diff):
        print(str(file_init.replace(".txt","")))
        if not file_init.endswith(".DS_Store"):
            matrix = np.loadtxt(str(input_direction + diff+"/"+ str(file_init)))
            print(matrix)
            matrix = np.where(matrix>0,1,matrix)
            matrix = np.where(matrix==0,0,matrix)
            matrix = np.where(matrix<0,-1,matrix)
            print(matrix)

            np.savetxt(output_direction_feature + diff + "/" + str(file_init),matrix,delimiter="  ")
            
            plot_density(matrix,output_direction_image + diff + "/" + str(file_init.replace(".txt","")),str(file_init.replace(".txt","")),"bwr", vmin=None, vmax=None)
            

            

def main():
    feature(path+"feature/task1/",path + "feature/task2/",path + "image/task2/","diff_t")
    feature(path+"feature/task1/",path + "feature/task2/",path + "image/task2/","diff_p")
    feature(path+"feature/task1/",path + "feature/task2/",path + "image/task2/","diff_v")

if __name__ == "__main__":main()
