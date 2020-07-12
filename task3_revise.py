import sys
# path = "/home/s1810235/part_time/PtNafion/"
path = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion/input/2020-06-17-NewRequest/txt/"

sys.path.insert(0, path)
from plot import plot_density
import os
import numpy as np
import glob
from copy import copy, deepcopy
from Nafion_constant import *
os.makedirs(path+ "image/task3",exist_ok=True)
os.makedirs(path+ "feature/task3",exist_ok=True)


def convert_bulk_bkg(output_feature,output_image,p,sub_square_size):
    morph_dir = path + p +"/morphology" 
    list_matrix = []
    bkg_thresh = 0.0008
    morph_files = [f for f in os.listdir(morph_dir) if f.endswith('.txt')]
    for morph_file in morph_files:  
        dfile = morph_dir+"/"+morph_file
        matrix = np.loadtxt(dfile)
        print(matrix)

        # matrix = np.ma.array(matrix, mask=np.isnan(matrix)) # Use a mask to mark the NaNs
        # print("IS NAN", np.count_nonzero(np.isnan(matrix)))

        matrix = np.where(matrix > bkg_thresh,"bulk","bkg")
        matrix,matrix_transform_to_plot = morph(matrix,sub_square_size)
        np.savetxt(path + output_feature + "bulk_label_" + p +str(morph_file) ,matrix,fmt="%s",delimiter="  ")
        np.savetxt(path + output_feature + "num_bulk_label"+ p + str(morph_file),matrix_transform_to_plot,fmt="%s",delimiter="  ")

        # matrix_transform_to_plot = transform_to_plot(matrix)
        matrix_transform_to_plot = matrix_transform_to_plot.astype(np.float)

        # vmin, vmax = get_vmin_vmax_diff(feature=feature)
        plot_density(values=matrix_transform_to_plot, 
            save_at=path + output_image + "bulk_label_"+p + str(morph_file.replace(".txt",".pdf")),
            title=str(dfile.replace(path,"") + "\nbackground threshold of morphology: {0}".format(bkg_thresh)) ,
            cmap_name="bwr", vmin=None, vmax=None)
        # break

    # elif(np.max(matrix)>4):
    #     matrix = np.where(matrix > 4 , "bulk","bkg" )
    #     matrix,matrix_transform_to_plot = morph(matrix,sub_square_size)
    #     np.savetxt(path + output_feature + "bulk_label_" + p  +str(file_init) ,matrix,fmt="%s",delimiter="  ")
    #     np.savetxt(path + output_feature + "num_bulk_label"+ p + str(file_init),matrix_transform_to_plot,fmt="%s",delimiter="  ")
    #     matrix_transform_to_plot = matrix_transform_to_plot.astype(np.float)
    #     plot_density(matrix_transform_to_plot, path + output_image + "bulk_label_"+p + str(file_init.replace(".txt","")),str(file_init.replace(".txt","")) ,  cmap_name="bwr", vmin=None, vmax=None)

                                      

def transform_to_plot(matrix):    
    matrix = np.where(matrix == "bulk", 1 , matrix)
    matrix = np.where(matrix == "bkg", -1 , matrix)
    matrix = np.where(matrix == "surf", 0 , matrix)
    
    return matrix


def check_feature(matrix,r,c,sub_square_size):
    if matrix[r,c]=="bkg":
        return "bkg"
    f1 = lambda x, y, min_value: x - y if x - y > min_value else min_value
    f2 = lambda x, y, max_value: x + y + 1 if x + y + 1 < max_value else max_value

    north = f1(r, sub_square_size, 0)
    west = f1(c, sub_square_size, 0)
    south = f2(r, sub_square_size, size_x)
    east = f2(c, sub_square_size, size_y)

    env = matrix[north:south,west:east]

    idx = np.nan
    if matrix[r,c]=="bulk":
        if "bkg" in env:
            print("SURF",r,c)
            idx = "surf"
        else:
            idx = "bulk"    
    return idx
    

def morph(elem,sub_square_size):
    # matrix_duplicate = elem
    # matrix_transform_to_plot = elem

    matrix_duplicate = copy(elem)
    matrix = copy(elem)
    
    print(matrix_duplicate)
    for row in range(matrix.shape[0]):
        for cell in range(matrix.shape[1]):
                matrix_duplicate[row,cell] = check_feature(matrix,row,cell,sub_square_size)
            
    print(matrix_duplicate)
    matrix_transform_to_plot = transform_to_plot(matrix_duplicate)
    print(matrix_transform_to_plot)
            
    return matrix_duplicate,matrix_transform_to_plot

def main():
    convert_bulk_bkg("feature/task3/","image/task3/","CCM-Nafion",1)
    convert_bulk_bkg("feature/task3/","image/task3/","CCMcenter",1)
    
    

    
    
        

if __name__ == "__main__":main()