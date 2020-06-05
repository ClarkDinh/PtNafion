import sys
path = "/home/s1810235/part_time/PtNafion/"
sys.path.insert(0, path)
from plot import plot_density
from plot import joint_plot_fill
from task3 import convert_bulk_bkg
import matplotlib.pyplot as plt
import numpy as np
import os 
os.makedirs(path +'image/task5/section2',exist_ok=True)

file_path = ["CCM-Nafion", "CCMcenter"]
for i in range(len(file_path)):
    for feature in os.listdir(path + file_path[i] + "/"):
        # if feature == "morphology" :
            if not feature.endswith(".DS_Store"):
                for file_init in os.listdir(path + file_path[i]+ "/" + str(feature)+ "/"):
                    if not file_init.startswith("._"):
                        if file_init.endswith(".txt"):
                            print(file_path[i] + str(feature)+ "/" + file_init)
                            temp = np.loadtxt(path  +file_path[i] + "/" + str(feature)+ "/" +file_init)
                            plt.hist(temp)
                            plt.savefig(path + 'image/task5/section2/{}_{}.pdf'.format(file_path[i],file_init.replace(".txt","")))
                            print("_"*30)
            
                





