import sys, os, re, math, time, multiprocessing, ntpath
from functools import partial

# cuda = torch.device('cuda') 
# path = "/home/s1810235/part_time/PtNafion/"
path = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion/input/2020-06-17-NewRequest/txt/"

sys.path.insert(0, path)
from plot import plot_density
# from opt_GMM_2 import *
# from dmin import *
# from constant import *
import numpy as np
from itertools import combinations
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from contextlib import contextmanager


os.makedirs(path + "feature/task4/",exist_ok=True)
os.makedirs(path + "image/task4/",exist_ok=True)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def bulk2surf(bulk_pos,surf_poses):
    distances = pairwise_distances(bulk_pos.reshape(1, -1), surf_poses).ravel()
    min_dist = np.min(distances)
    return min_dist

def get_file_name(string):
    return ("dmin"+str(re.findall(r'num_bulk_label(.*?).txt', string)))

def get_basename(filename):
    head, tail = ntpath.split(filename)
    basename = os.path.splitext(tail)[0]
    return tail

def dmin_v3(path_,filename,feature_dir,image_dir):
    bulk_poses = [] # # Nguyen: bulk_poses index of bulk 
    surf_poses = [] # # Nguyen: surf_poses index of surface 
    matrix_distance = []
    matrix = np.loadtxt(path_ + filename)
    matrix = np.where(matrix==-1,np.nan,matrix)
    print(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] == 1:
                bulk_poses.append([i,j])
            elif matrix[i,j] == 0:
                surf_poses.append([i,j])

    bulk_poses = np.array(bulk_poses)
    surf_poses = np.array(surf_poses)
    
    with poolcontext(processes=3) as pool:
        result_min_dists = pool.map(partial(bulk2surf, surf_poses=surf_poses), bulk_poses)
    
    # all_array_of_distance = np.array(all_array_of_distance)
    # matrix_distance = np.min(all_array_of_distance,axis=-1)
    dmin_matrix = np.full(matrix.shape, -50.0)
    for idx, i in enumerate(bulk_poses):
        dmin_matrix[i[0],i[1]] = result_min_dists[idx]
    print(dmin_matrix)

    np.savetxt(feature_dir+get_basename(filename).replace("num_bulk_label", "dmin_"), dmin_matrix,delimiter="  ")
    # vì nan sẽ làm tính min max trong hàm plot bị lỗi, nên set nan thành 1 giá trị nào đó khác 0(surf) 
    # và âm thì nhìn thấy màu rõ hơn vì giá trị distance toàn là dương. 
    # test[np.isnan(test)] = -30 
    plot_density(values=dmin_matrix, save_at=image_dir, title=get_basename(filename).replace("num_bulk_label", "dmin_"), 
        cmap_name="jet", vmin=-50, vmax=50)


def main():
    list_file_name = [ 
    # "num_bulk_labelCCM-NafionADT5k1V_morphology.txt",
    # "num_bulk_labelCCM-NafionADT5k04V_morphology.txt",
    # "num_bulk_labelCCMcenterADT5k1V_morphology.txt",
    # "num_bulk_labelCCMcenterADT5k04V_morphology.txt",

    "num_bulk_labelCCMcenterFresh1V_morphology.txt",
    "num_bulk_labelCCMcenterFresh04V_morphology.txt",
    "num_bulk_labelCCM-NafionFresh1V_morphology.txt",
    "num_bulk_labelCCM-NafionFresh04V_morphology.txt",
    "num_bulk_labelCCMcenterADT15k1V_morphology.txt",
    "num_bulk_labelCCMcenterADT15k04V_morphology.txt",
    "num_bulk_labelCCM-NafionADT15k1V_morphology.txt",
    "num_bulk_labelCCM-NafionADT15k04V_morphology.txt",
    ]
    for filename in list_file_name:
        dmin_v3(path_=path + "feature/task3/",
            filename=filename,
            feature_dir=path + "feature/task4/", 
            image_dir=path+"image/task4/"+filename.replace(".txt",".pdf").replace("num_bulk_label", "dmin_"))

if __name__ == "__main__":main()



                                

                             

                        





    

