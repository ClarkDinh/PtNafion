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

from Nafion_constant import *
from redox_label import pos_neg_lbl_cvt

dmin_calc_dir = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion/fromQuan/v4_0526/feature/task4"

def get_dmin(fixP, fixT, fixV):
    dmin_file = "{0}/task4/dmin{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
    dmin_value = np.loadtxt(dmin_file).ravel()
    return dmin_value

def remove_nan(matrix1,matrix2):
    nan_of_matrix1 = np.argwhere(np.isnan(matrix1))
    nan_of_matrix2 = np.argwhere(np.isnan(matrix2))
    print(nan_of_matrix1, nan_of_matrix2)
    matrix1 = np.delete(matrix1,np.concatenate((nan_of_matrix1,nan_of_matrix2),axis=0))
    matrix2 = np.delete(matrix2,np.concatenate((nan_of_matrix1,nan_of_matrix2),axis=0))
    return matrix1,matrix2

def task1_struct_params_vs_dmin(task, fixP, fixT, fixV):

    # # get dmin
    dmin_value = get_dmin(fixP, fixT, fixV)

    # # get redox value
    for feature in Features:
        ft_file = myinput_dir+"/"+fixP+"/"+feature+"/"+fixT+fixV+"_"+feature+".txt"
        ft_val = np.loadtxt(ft_file).ravel()
        xlabel = "dmin"
        x, y = remove_nan(dmin_value,ft_val)
        min_x = np.min(x) 
        max_x = np.max(x)
        min_y = np.min(y) * 0.9 - 0.1
        max_y = np.max(y) * 1.1 + 0.1
        if feature == "morphology":
            ft_val = ft_val*1000 
            ylabel = feature+"(*10^3)"
            min_y = -1 
            max_y = 4 
        else:
            ylabel = feature

        start = time.time()

        print(min_x, max_x, min_y, max_y)
        print(x, y)

        save_at = "{0}/new_request/{1}/".format(result_dir,task)+feature+"_"+fixP+"_"+fixT+"_"+fixV+".pdf"

        try:
            joint_plot_fill(x=dmin_value, y=ft_val,
                xlabel=xlabel, ylabel=ylabel, save_at=save_at, 
                lbx=-2, ubx=40, lby=min_y, uby=max_y)
        except Exception as e:
            joint_plot(x=dmin_value, y=ft_val, 
                    xlabel=xlabel, 
                    ylabel=ylabel, 
                    xlim=[-2, 40], 
                    ylim=[min_y, max_y],
                    title=save_at.replace(result_dir, ""),
                    save_at=save_at)

        # joint_plot_fill(, , 
        #     str(name[0]), str(name[1]), path + "image/task5/joint_org/{}___{}.pdf".format(str(name[0]),str(name[1])) , min_x ,max_x,min_y,max_y )
        print("finish............" + "Time take:{}".format(time.time()-start))




def task2_deltaPT_vs_dmin(fixT, fixP, fixV, diff_state, task):
    final_state, init_state = diff_state # # here is diff voltage

    # # only for dt
    fix_val = "{0}{1}".format(fixP, fixV)

    # dminCCM-NafionADT15k04V_morphology.txt
    # dminCCM-NafionFresh04V_morphology.txt

    # # get dmin
    dmin_file = "{0}/task4/dmin{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
    dmin_value = np.loadtxt(dmin_file)

    # # get Pt-density value

    consider_Ft = "Pt-density"
    prefix_input = fix_val + consider_Ft
    diff_Ptdens = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
                    task, prefix_input, 
                    final_state, init_state)
    diff_PtO_val, is_diff_PtO_pos = pos_neg_lbl_cvt(inputfile=diff_Ptdens)


    save_at = "{0}/new_request/{1}/{2}_{3}___{4}_redox.pdf".format(result_dir,
        task, fix_val, final_state,  init_state) 

    joint_plot(x=dmin_value.ravel(), y=redox_label.ravel(), 
        xlabel="dmin_{0}{1}{2}".format(fixP, fixT, fixV), ylabel="redox_state", 
        xlim=[-2, 40], ylim=None,
        title=save_at.replace(result_dir, ""),
        save_at=save_at)



def main():
    # # task 1
    task = "task2_deltaPt" # task2_deltaPt

    if task == "task1_dmin_corr":
        fixV = "04V"
        for fixP in Positions:
            for fixT in Times:
                task1_struct_params_vs_dmin(task=task, fixP=fixP, fixT=fixT, fixV=fixV)

    if task == "task2_deltaPt":
        a = [dT_tunes, Positions, Voltages]
        combs = list(itertools.product(*a))
        for comb in combs:
            diff_T, fixP, fixV = comb
            fixT = diff_T[0]
            task2_deltaPT_vs_dmin(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_T, task=task)





if __name__ == "__main__":
    main()
