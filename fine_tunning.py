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
from redox_label import redox_state_lbl
import copy

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

def task1_get_ftfile_saveat(task, fixP, fixT, fixV, feature):
    ft_file = myinput_dir+"/"+fixP+"/"+feature+"/"+fixT+fixV+"_"+feature+".txt"
    save_at = "{0}/new_request/{1}/".format(result_dir,task)+feature+"_"+fixP+"_"+fixT+"_"+fixV+".pdf"
    save_at_sb = "{0}/new_request/{1}_sb/".format(result_dir,task)+feature+"_"+fixP+"_"+fixT+"_"+fixV+".pdf"
    
    return ft_file, save_at, save_at_sb

def task1_struct_params_vs_dmin(task, fixP, fixT, fixV):

    # # get dmin
    dmin_value = get_dmin(fixP, fixT, fixV)
    xlabel = "dmin"
    xlim=[-2, 90]


    feature = "Pt-density"
    ft_file, save_at, save_at_sb = task1_get_ftfile_saveat(task, fixP, fixT, fixV, feature)
    ft_val = np.loadtxt(ft_file).ravel()
    dmin_value_copy = copy.copy(dmin_value)
    x, y = remove_nan(dmin_value_copy,ft_val)
    if fixT != "ADT5k":
        y = y * 10000
    joint_plot_2(x=x, y=y, xlabel=xlabel, ylabel=feature, 
        xlim=xlim, ylim=[-2, 50], 
        title=save_at.replace(result_dir, ""), save_at=save_at)
    # joint_plot(x=x, y=y, xlabel=xlabel, ylabel=feature, 
    #     xlim=xlim, ylim=[-2, 50], 
    #     title=save_at.replace(result_dir, ""),
    #      save_at=save_at_sb)

    # feature = "Pt-valence"
    # ft_file, save_at, save_at_sb = task1_get_ftfile_saveat(task, fixP, fixT, fixV, feature)
    # ft_val = np.loadtxt(ft_file).ravel()
    # dmin_value_copy = copy.copy(dmin_value)
    # x, y = remove_nan(dmin_value_copy,ft_val)
    # # if fixT != "ADT5k":
    # #     y = y * 10000
    # # ylim = [np.min(y), np.max(y)]
    # ylim = [-0.1, 1.6]

    # joint_plot_2(x=x, y=y, xlabel=xlabel, ylabel=feature, 
    #     xlim=xlim, ylim=ylim, 
    #     title=save_at.replace(result_dir, ""), save_at=save_at)
    # joint_plot(x=x, y=y, xlabel=xlabel, ylabel=feature, 
    #     xlim=xlim, ylim=ylim, 
    #     title=save_at.replace(result_dir, ""),
    #      save_at=save_at_sb)


    # feature = "Pt-O"
    # ft_file, save_at, save_at_sb = task1_get_ftfile_saveat(task, fixP, fixT, fixV, feature)
    # ft_val = np.loadtxt(ft_file).ravel()
    # dmin_value_copy = copy.copy(dmin_value)
    # x, y = remove_nan(dmin_value_copy,ft_val)
    # # ylim = [np.min(y), np.max(y)]
    # ylim = [-0.1, 2.1]

    # joint_plot_2(x=x, y=y, xlabel=xlabel, ylabel=feature, 
    #     xlim=xlim, ylim=ylim, 
    #     title=save_at.replace(result_dir, ""), save_at=save_at)
    # joint_plot(x=x, y=y, xlabel=xlabel, ylabel=feature, 
    #     xlim=xlim, ylim=ylim, 
    #     title=save_at.replace(result_dir, ""),
    #      save_at=save_at_sb)


    # feature = "Pt-Pt"
    # ft_file, save_at, save_at_sb = task1_get_ftfile_saveat(task, fixP, fixT, fixV, feature)
    # ft_val = np.loadtxt(ft_file).ravel()
    # dmin_value_copy = copy.copy(dmin_value)
    # x, y = remove_nan(dmin_value_copy,ft_val)

    # # ylim = [np.min(y), np.max(y)]
    # ylim = [-0.1, 13]

    # joint_plot_2(x=x, y=y, xlabel=xlabel, ylabel=feature, 
    #     xlim=xlim, ylim=ylim, 
    #     title=save_at.replace(result_dir, ""), save_at=save_at)
    # joint_plot(x=x, y=y, xlabel=xlabel, ylabel=feature, 
    #     xlim=xlim, ylim=ylim, 
    #     title=save_at.replace(result_dir, ""),
    #      save_at=save_at_sb)
    # try:
    #     joint_plot_fill(x=dmin_value, y=ft_val,
    #         xlabel=xlabel, ylabel=ylabel, save_at=save_at, 
    #         lbx=-2, ubx=40, lby=min_y, uby=max_y)
    # except Exception as e:


def task2_deltaPT_vs_dmin(fixT, fixP, fixV, diff_state, task):
    final_state, init_state = diff_state # # here is diff voltage

    # # only for dt
    fix_val = "{0}{1}".format(fixP, fixV)

    # dminCCM-NafionADT15k04V_morphology.txt
    # dminCCM-NafionFresh04V_morphology.txt

    # # get dmin
    dmin_file = "{0}/task4/dmin{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
    dmin_value = np.loadtxt(dmin_file)
    xlim=[-2, 90]

    # # get Pt-density value

    consider_Ft = "Pt-density"
    prefix_input = fix_val + consider_Ft
    diff_Ptdens = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
                    "diff_t", prefix_input, 
                    final_state, init_state)
    diff_Ptdens_val, is_diff_Ptdens_pos = pos_neg_lbl_cvt(inputfile=diff_Ptdens)

    diff_Ptdens_val_fix = 0-diff_Ptdens_val
    # # 2d image
    save_at = "{0}/new_request/{1}/img_{2}_{3}___{4}.pdf".format(result_dir,
        task, fix_val, final_state,  init_state) 
    plot_density(values=diff_Ptdens_val_fix, save_at=save_at,  
        # cmap_name="Oranges", vmin=None, vmax=None, is_save2input=False
        cmap_name="bwr", vmin=-0.004, vmax=0.004
        )
# plot_density(matrix_transform_to_plot, path + output_image + "bulk_label_"+p + str(file_init.replace(".txt","")),str(file_init.replace(".txt","")) ,  )



    save_at = "{0}/≈/{1}/{2}_{3}___{4}_redox.pdf".format(result_dir,
        task, fix_val, final_state,  init_state) 
    x, y = remove_nan(dmin_value.ravel(),diff_Ptdens_val_fix.ravel())

    min_y = np.min(y) * 0.9 
    max_y = np.max(y) * 1.1 
    joint_plot(x=x, y=y, 
        xlabel="dmin", ylabel=consider_Ft, 
        xlim=xlim, ylim=[min_y, max_y],
        title=save_at.replace(result_dir, ""),
        save_at=save_at)

 
def task3_reaction_mode(fixT, fixP, fixV, diff_state):
    # ADT5k1VPt-density_CCM-Nafion____CCMcenter.pdf
    final_state, init_state = diff_state # # here is diff voltage

    task = "diff_t"
    fix_val = "{0}{1}".format(fixP, fixV)
    
    consider_Ft = "Pt-O"
    prefix_input = fix_val + consider_Ft
    diff_PtO = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
                    task, prefix_input, 
                    final_state, init_state)
    diff_PtO_val, is_diff_PtO_pos = pos_neg_lbl_cvt(inputfile=diff_PtO)


    consider_Ft = "Pt-valence"
    prefix_input = fix_val + consider_Ft
    diff_PtVal = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
                    task, prefix_input, 
                    final_state, init_state)
    diff_PtVal_val, is_diff_PtVal_pos = pos_neg_lbl_cvt(inputfile=diff_PtVal)



    consider_Ft = "Pt-Pt"
    prefix_input = fix_val + consider_Ft
    diff_PtPt = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
                    task, prefix_input, 
                    final_state, init_state)

    diff_PtPt_val, is_diff_PtPt_pos = pos_neg_lbl_cvt(inputfile=diff_PtPt)

    # all shape checked
    # print ("is_diff_PtDens_pos", diff_PtDens_val)
    # print ("is_diff_PtVal_pos", diff_PtVal_val)
    # print ("is_diff_PtPt_pos", diff_PtPt_val)


    redox_states = redox_state_lbl(is_diff_PtO_pos=is_diff_PtO_pos, 
        is_diff_PtVal_pos=is_diff_PtVal_pos, 
        is_diff_PtPt_pos=is_diff_PtPt_pos)


    redox_sum = sum(redox_states)
    redox_sum = redox_sum.astype(float)
    redox_sum[redox_sum==3] = np.nan
    redox_sum[redox_sum==4] = np.nan
    redox_sum[redox_sum==6] = np.nan

    # redox_sum = np.where(redox_sum==3, np.nan, redox_sum)
    # redox_sum = np.where(redox_sum==4, np.nan, redox_sum)
    # redox_sum = np.where(redox_sum==6, np.nan, redox_sum)

    # # to plot_density
    vmin=1
    vmax=8

    cmap_name="jet"
    prefix = "{0}/new_request/task3/{1}_{2}___{3}".format(result_dir,
         fix_val, final_state, init_state) 
    save_at = prefix + ".pdf"
    plot_density(values=redox_sum, save_at=save_at,  # diff_PtDens_lbl
        cmap_name=cmap_name, vmin=vmin, vmax=vmax)

    # # joint dmin vs redox

    # # get dmin
    dmin_file = "{0}/task4/dmin{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
    dmin_value = np.loadtxt(dmin_file)

    # # get redox value
    redox_file = "{0}/redox/{1}/{2}_{3}___{4}.txt".format(myinput_dir,
        task, fix_val, final_state, init_state)
    redox_label = np.loadtxt(redox_file)
    # np.where(redox_label==3,0.0,redox_label)
    # np.where(redox_label==4,0.0,redox_label)
    # np.where(redox_label==6,0.0,redox_label)

    save_at = "{0}/new_request/task3/dmin_redox_{1}_{2}___{3}_redox.pdf".format(result_dir,
    fix_val, final_state,  init_state) 
    x, y = remove_nan(dmin_value.ravel(),redox_sum.ravel())



    joint_plot(x=x, y=y, 
        xlabel="dmin_{0}{1}{2}".format(fixP, fixT, fixV), ylabel="redox_state", 
        xlim=[-2, 40], ylim=[0, 8.5],
        title=save_at.replace(result_dir, ""),
        save_at=save_at)


def main():
    # # task 1
    task = "task3" # task1_dmin_corr, task2_deltaPt, task3

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

    if task == "task3":
        a = [dT_tunes, Positions, Voltages]
        combs = list(itertools.product(*a))
        for comb in combs:
            diff_T, fixP, fixV = comb
            fixT = diff_T[0]
            task3_reaction_mode(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_T)




if __name__ == "__main__":
    main()
