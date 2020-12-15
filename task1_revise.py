import sys, copy
# path = "/home/s1810235/part_time/PtNafion/"
path = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion/input/2020-06-17-NewRequest/txt/"
sys.path.insert(0, path)
from plot import *
# from opt_GMM_2 import *
import numpy as np
import os, time
from Nafion_constant import *
# dir_list = os.listdir(path) 
# print(dir_list)



def diff_t(p,v,t1,t2,task):
    sub_path = path + p +"/" 
    count = 0
    for feature in os.listdir(sub_path):
        if not feature.endswith(".DS_Store"):
            if(task=="task1"):
                Pt1 = np.array(np.loadtxt(sub_path+ str(feature)+"/" + t1 + v + "_" + str(feature)+".txt"))
                Pt2 = np.array(np.loadtxt(sub_path+ str(feature)+"/" + t2 + v + "_" + str(feature)+".txt"))
                Pt1_t2 = np.subtract(Pt1,Pt2)
                # print(Pt1_t2)
                os.makedirs(path + "0916_image/task1/" + "diff_t",exist_ok=True)

                vmin, vmax = get_vmin_vmax_diff(feature=feature)
                print (feature, vmin, vmax)
                # vmin, vmax = None, None'
                print ("size before: ", Pt1_t2.shape)
                Pt1_t2_copy =  copy.copy(Pt1_t2)
                Pt1_t2_copy = np.delete(Pt1_t2_copy, index2remove, 1)  # delete second column of C
                print ("size after: ", Pt1_t2_copy.shape)
                
                plot_density(values=Pt1_t2_copy,
                    save_at=path+"0916_image/task1/"+"diff_t/"+"{Params}_{feature1}____{feature2}.pdf".format(Params=str(p+v+feature),feature1=str(t1),feature2 = str(t2)),
                    cmap_name="bwr",
                    title="{Params}_{feature1}____{feature2}".format(Params=str(p+v+feature),feature1=str(t1),feature2=str(t2)),
                    vmin=vmin, vmax=vmax)
                localtime = time.localtime()
                result = time.strftime("%I:%M:%S %p", localtime)
                print(result, end="", flush=True)
                print("\r", end="", flush=True)
                
                
                os.makedirs(path + "feature/task1/" + "diff_t" ,exist_ok=True)
                np.savetxt(path + "feature/task1/" + "diff_t/" + "{Params}_{feature1}____{feature2}.txt".format(Params=str(p + v + feature),feature1=str(t1),feature2=str(t2)),Pt1_t2,delimiter="  ")
            else:
                # count +=1 
                Pt1 = np.ravel(np.loadtxt(sub_path+ str(feature)+"/" + t1 + v + "_" + str(feature)+".txt"))
                Pt2 = np.ravel(np.loadtxt(sub_path+ str(feature)+"/" + t2 + v + "_" + str(feature)+".txt"))
                ######task5_section3######
                min_x = np.min(Pt1) ######x tmp[0],y=temp[1]
                max_x = np.max(Pt1)
                min_y = np.min(Pt2) ######x tmp[0],y=temp[1]
                max_y = np.max(Pt2)
                start = time.time()
                os.makedirs(path + "image/task5/joint_diff/",exist_ok=True)
                print(feature)
                print("processing.........")
                joint_plot_fill(Pt1, Pt2, "{}_{}_{}_{}".format(feature,p,t1,v), "{}_{}_{}_{}".format(feature,p,t2,v), path + "image/task5/joint_diff/{}___{}.pdf".format("{}_{}_{}_{}".format(feature,p,t1,v), "{}_{}_{}_{}".format(feature,p,t2,v)) , min_x ,max_x,min_y,max_y )
                print("finish............" + "Time take:{}".format(time.time()-start)+"to run"+"file num:{}".format(count)) 
                ######task5######
                
                
def diff_p(t,v,p1,p2,task):
    sub_path_p1 = path + p1 + "/"
    sub_path_p2 = path + p2 + "/"
    
    for feature in os.listdir(sub_path_p1):
        if not feature.endswith(".DS_Store"):
            if(task=="task1"):
                Pp1 = np.array(np.loadtxt(sub_path_p1 + str(feature) + "/" + t + v + "_" + str(feature)+ ".txt"))
                Pp2 = np.array(np.loadtxt(sub_path_p2 + str(feature) + "/" + t + v + "_" + str(feature)+ ".txt"))
                Pp1_p2 = np.subtract(Pp1,Pp2)
                os.makedirs(path + "image/task1/" + "diff_p",exist_ok=True)
                
                vmin, vmax = get_vmin_vmax_diff(feature=feature)

                plot_density(values=Pp1_p2,
                    save_at=path + "image/task1/" + "diff_p/"+"{Param}_{feature1}____{feature2}.pdf".format(Param = str(t + v + feature),feature1=str(p1),feature2=str(p2)),
                    title="{Param}_{feature1}____{feature2}".format(Param = str(t + v + feature),feature1 = str(p1),feature2 = str(p2)),
                    cmap_name="bwr", vmin=vmin, vmax=vmax)
                os.makedirs(path + "feature/task1/" + "diff_p",exist_ok=True)
                np.savetxt(path + "feature/task1/" + "diff_p/"+"{Param}_{feature1}____{feature2}.txt".format(Param = str(t + v + feature),feature1=str(p1),feature2=str(p2)),
                    Pp1_p2,delimiter="  ")
                ######task5_section3######
            else:
                Pp1 = np.ravel(np.loadtxt(sub_path_p1 + str(feature) + "/" + t + v + "_" + str(feature)+ ".txt"))
                Pp2 = np.ravel(np.loadtxt(sub_path_p2 + str(feature) + "/" + t + v + "_" + str(feature)+ ".txt"))
                min_x = np.min(Pp1) ######x tmp[0],y=temp[1]
                max_x = np.max(Pp1)
                min_y = np.min(Pp2) ######x tmp[0],y=temp[1]
                max_y = np.max(Pp2)
                start = time.time()
                os.makedirs(path + "image/task5/joint_diff/",exist_ok=True)
                print("processing.........")
                joint_plot_fill(Pp1, Pp2, "{}_{}_{}_{}".format(feature,p1,t,v), "{}_{}_{}_{}".format(feature,p2,t,v), path + "image/task5/joint_diff/{}___{}.pdf".format("{}_{}_{}_{}".format(feature,p1,t,v), "{}_{}_{}_{}".format(feature,p2,t,v)) , min_x ,max_x,min_y,max_y )
                print("finish............" + "Time take:{}".format(time.time()-start)+"to run"+"file num:{}".format(count))

def diff_v(t,p,v1,v2,task):
    sub_path = path + p + "/"
    count = 0

    for feature in os.listdir(sub_path):
        if not feature.endswith(".DS_Store"):
            if(task=="task1"):
                Pv1 = np.array(np.loadtxt(sub_path + str(feature) + "/" + t + v1 + "_" + str(feature)+ ".txt"))
                Pv2 = np.array(np.loadtxt(sub_path + str(feature) + "/" + t + v2 + "_" + str(feature)+ ".txt"))
                Pv1_2 = np.subtract(Pv1,Pv2)
                os.makedirs(path +"image/task1/" + "diff_v",exist_ok=True)
                vmin, vmax = get_vmin_vmax_diff(feature=feature)
                print (feature, vmin, vmax)
                # vmin, vmax = None, None'
                print ("size before: ", Pv1_2.shape)
                Pv1_2_copy =  copy.copy(Pv1_2)
                Pv1_2_copy = np.delete(Pv1_2_copy, index2remove, 1)  # delete second column of C
                print ("size after: ", Pv1_2_copy.shape)


                plot_density(values=Pv1_2_copy,
                    save_at=path + "image/task1/" + "diff_v/"+"{Param}_{feature1}___{feature2}.pdf".format(Param = str(t + p + feature),feature1 = str(v1),feature2 = str(v2)),
                    title="{Param}_{feature1}____{feature2}".format(Param = str(t + p + feature),feature1 = str(v1),feature2 = str(v2)),
                    cmap_name="bwr", vmin=vmin, vmax=vmax)
                os.makedirs(path +"feature/task1/" + "diff_v",exist_ok=True)
                np.savetxt(path + "feature/task1/" + "diff_v/"+"{Param}_{feature1}___{feature2}.txt".format(Param = str(t + p + feature),feature1 = str(v1),feature2 = str(v2)),
                    Pv1_2,delimiter="  ")
                

def main():
    # # #Different of parameter T
    # diff_t("CCM-Nafion","1V","ADT15k","Fresh","task1")
    # diff_t("CCM-Nafion","04V","ADT15k","Fresh","task1")
    # diff_t("CCMcenter","1V","ADT15k","Fresh","task1")
    # diff_t("CCMcenter","04V","ADT15k","Fresh","task1")

    # #Different of paremeter V
    diff_v("Fresh","CCMcenter","1V","04V","task1")
    diff_v("Fresh","CCM-Nafion","1V","04V","task1")
    diff_v("ADT15k","CCMcenter","1V","04V","task1")
    diff_v("ADT15k","CCM-Nafion","1V","04V","task1")

    # # Different of parameter p
    # diff_p("ADT15k","1V","CCM-Nafion","CCMcenter","task1")
    # diff_p("ADT15k","04V","CCM-Nafion","CCMcenter","task1")
    # diff_p("Fresh","1V","CCM-Nafion","CCMcenter","task1")
    # diff_p("Fresh","04V","CCM-Nafion","CCMcenter","task1")

    # # ignore in new_request
    # diff_t("CCM-Nafion","1V","Fresh","ADT5k","task1")
    # diff_t("CCM-Nafion","04V","Fresh","ADT5k","task1")
    # diff_t("CCM-Nafion","1V","ADT5k","ADT15k","task1")
    # diff_t("CCM-Nafion","04V","ADT5k","ADT15k","task1")
    # diff_t("CCMcenter","1V","Fresh","ADT5k","task1")
    # diff_t("CCMcenter","04V","Fresh","ADT5k","task1")
    # diff_t("CCMcenter","1V","ADT5k","ADT15k","task1")
    # diff_t("CCMcenter","04V","ADT5k","ADT15k","task1")
    # # diff_p("ADT5k","1V","CCM-Nafion","CCMcenter","task1")
    # # diff_p("ADT5k","04V","CCM-Nafion","CCMcenter","task1")
    # # diff_v("ADT5k","CCMcenter","04V","1V","task1")
    # # diff_v("ADT5k","CCM-Nafion","04V","1V","task1")




if __name__ == "__main__":main()
