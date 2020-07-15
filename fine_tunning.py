import sys
path = "/home/s1810235/part_time/PtNafion/"
sys.path.insert(0, path)
from plot import *
from opt_GMM_2 import *

from redox_label import pos_neg_lbl_cvt, redox_state_lbl
import numpy as np
import os 
from itertools import combinations
import re 
import time

from Nafion_constant import *
# from opt_GMM_2 import plot_joinmap
import copy
import pandas as pd
from sklearn import mixture

# dmin_calc_dir = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion/fromQuan/v4_0526/feature/task4"
# new_inputdir = maindir + "/input/2020-06-17-NewRequest/txt"
# input_dir = "{}/input/2020-06-17-NewRequest/txt".format(maindir)


def get_dmin(fixP, fixT, fixV):
    dmin_file = "{0}/feature/task4/dmin_{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
    dmin_value = np.loadtxt(dmin_file).ravel()
    return dmin_value

def task1_get_ftfile_saveat(task, fixP, fixT, fixV, feature):
    ft_file = input_dir+"/"+fixP+"/"+feature+"/"+fixT+fixV+"_"+feature+".txt"
    save_at = "{0}/new_request_3/{1}/".format(result_dir,task)+feature+"_"+fixP+"_"+fixT+"_"+fixV+".pdf"
    save_at_sb = "{0}/new_request_3/{1}_sb/".format(result_dir,task)+feature+"_"+fixP+"_"+fixT+"_"+fixV+".pdf"
    
    return ft_file, save_at, save_at_sb

def main_ax_plot(x,y,ax,xlbl,ylbl,color,ax_type):
    x1 = pd.Series(x, name=xlbl)
    x2 = pd.Series(y, name=ylbl)


    
    colors = "jet"
    # label_patch = mpatches.Patch(
    #     color=sns.color_palette(colors),
    #     # label=lbl
    #     )
    if ax_type == "kde":
        ax.scatter(x, y, color="grey",# label=state, 
                alpha=0.1, s=1, # marker="o",
                # linewidths=0.1,
                edgecolors=None)
        ax = sns.kdeplot(x1, x2,
             #joint_kws={"colors": "black", "cmap": None, "linewidths": 0.5},
             cmap=colors,
             shade=False, shade_lowest=True,
             n_levels=10,
             fontsize=10,
             ax=ax,
             # vertical=True
             )

        
        # x_c = np.linspace(np.min(x)-0.1, np.max(x)+0.1, 100)
        # y_c = np.linspace(np.min(y)-0.5, np.max(y)+0.5, 100)
        # X, Y = np.meshgrid(x_c, y_c)
        # XX = np.array([X.ravel(), Y.ravel()]).T
        # gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(np.array([x,y]))
        # Z = - gmm.score_samples(XX)
        # Z = Z.reshape(X.shape)
    
        # xmin = np.min(x)
        # xmax = np.max(x)
        # ymin = np.min(y)
        # ymax = np.max(y)
        # X, Y = np.mgrid[xmin:xmax:10j, ymin:ymax:10j]
        # positions = np.vstack([X.ravel(), Y.ravel()])
        # values = np.vstack([x, y])
        # kernel = stats.gaussian_kde(values, bw_method="silverman")
        # Z = np.reshape(kernel(positions).T, X.shape)
        # ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
        #    # extent=[xmin, xmax, ymin, ymax]
        #    )
        # ax.plot(x, y, 'k.', markersize=1)

    elif ax_type == "scatter":
        ax.scatter(x, y, color=color,# label=state, 
                alpha=0.6, s=2, # marker="o",
                # linewidths=0.1,
                edgecolors=None)
    elif ax_type == "violin":
        ax = sns.violinplot(x=x, y=y, color=color, orient="h", ax=ax,
            cut=0, order=[1, 2, 3, 4, 5, 6, 7, 8],
            scale="count", # {“area”, “count”, “width”}
            gridsize=50, linewidth=1.0,
            inner="quartile" # {“box”, “quartile”, “point”, “stick”}, 
            )
    elif ax_type == "candle":
        df = pd.DataFrame(columns=[xlbl,ylbl])
        df[xlbl] = x
        df[ylbl] = y

        dmin = copy.copy(x)
        glbl_array = copy.copy(y)

        group_labels = [1,2,3,4,5,6,7,8]
        
        # df_plt = pd.DataFrame(columns=group_labels)
        # for glbl in group_labels:
        #     idxes = np.where(glbl_array==glbl)[0]
        #     df_plt[glbl] = dmin[idxes]
        pal = sns.color_palette(n_colors=8)
        import ptitprince as pt

        ort = "h"
        # ax=pt.half_violinplot(x=xlbl,y=ylbl,data=df_plt, palette=pal,
        #   bw=.15, cut=0.,scale="area", width=.6, inner=None,
        #   orient = "h", ax=ax)

        for glbl in group_labels:
            df_plt = pd.DataFrame(columns=[xlbl,ylbl])
            idx = np.where(glbl_array==glbl)[0]
            df_plt[xlbl] = x[idx]
            df_plt[ylbl] = y[idx]
            mean = np.mean(x[idx])
            ax.boxplot(x=x[idx], vert=False, notch=True, sym='rs', positions=[glbl],
                # whiskerprops={'linewidth':2, "zorder":10},
                widths=0.5, meanline=True, 
                showfliers=True, showbox=True, showmeans=True)
            ax.text(mean, glbl, mean,
                horizontalalignment='center', size='x-small', 
                color='black', weight='semibold')

            # ax=sns.stripplot(x=xlbl, y=ylbl, data=df_plt, color="blue",#palette=pal, 
            #    edgecolor="white", size=2, jitter=1, zorder=0, 
            #    orient=ort, ax=ax, positions=[glbl])

            # ax=sns.boxplot(x=xlbl, y=ylbl, data=df_plt, color="black", 
            #        width=.15, zorder=10, showcaps=True,
            #        boxprops={'facecolor':'none', "zorder":10}, showfliers=False,
            #        whiskerprops={'linewidth':2, "zorder":10},
                   # saturation=1, orient=ort, ax=ax, positions=[glbl])
        # candlestick_ochl(ax, df_plt.values, width=4, colorup='g', colordown='r', alpha=1)

    # ax.legend(handles=label_patch, loc='upper right')
    return ax

def plot_joinmap(df, selected_inst, xlbl, ylbl, color="blue",
                xlbl_2fig=None, ylbl_2fig=None, 
                save_at=None, 
                is_gmm_xhist=True, is_gmm_yhist=True, 
                means=None, weight=None, cov_matrix=None, 
                n_components=None, xlim=None, ylim=None,
                main_ax_type="scatter"):
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    main_ax = fig.add_subplot(grid[1:, :-1])
    y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
    

    sns.set_style('ticks')

    if selected_inst is None:
        selected_inst = df.index
    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # plot kde contour of each components
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    main_ax = main_ax_plot(x=df.loc[selected_inst, xlbl].values,y=df.loc[selected_inst, ylbl].values,
            ax=main_ax,xlbl=xlbl,ylbl=ylbl,color=color,ax_type=main_ax_type)
    main_ax.axhline(y=0.0, linestyle="-.", c="red", xmin=xlim[0], xmax=xlim[1])
    if xlbl_2fig is not None and ylbl_2fig is not None:
        is_add_latex = True # if use in Latex format

        if is_add_latex:
            main_ax.set_xlabel(r"{0}".format(xlbl_2fig), **axis_font)
            main_ax.set_ylabel(r"{0}".format(ylbl_2fig), **axis_font)
        else:
            main_ax.set_xlabel(xlbl_2fig, **axis_font)
            main_ax.set_ylabel(ylbl_2fig, **axis_font)
   
    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # plot gmm_xhist
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    if is_gmm_xhist:
        n, bins, patches = x_hist.hist(df.loc[selected_inst, xlbl].values, 200, normed=True, 
            facecolor=color, alpha=0.3, orientation="horizontal")
    else:
        sns.distplot(df.loc[selected_inst, xlbl].values,  color=color, ax=x_hist, norm_hist=False)
    print ("Done plot is_gmm_xhist")

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # plot gmm_yhist
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    if is_gmm_yhist:
        n, bins, patches = y_hist.hist(df.loc[selected_inst, ylbl].values, 200, normed=True, 
            facecolor=color, alpha=0.3, orientation="vertical")
    else:
        sns.distplot(df.loc[selected_inst, ylbl].values,  color=color, ax=y_hist, 
            vertical=True, norm_hist=False)

    main_ax.tick_params(axis='both', labelsize=10)
    if xlim is not None:
        main_ax.set_xlim(xlim)
        x_hist.set_xlim(xlim)

    if ylim is not None:
        main_ax.set_ylim(ylim)
        y_hist.set_ylim(ylim)

    plt.title(save_at.replace(result_dir, "").replace("/new_request_3/", "").replace(".pdf", ""))
    # plt.legend()
    plt.setp(x_hist.get_xticklabels(), visible=False)
    plt.setp(y_hist.get_yticklabels(), visible=False)
    plt.tight_layout(pad=1.1)
    makedirs(save_at)
    plt.savefig(save_at, transparent=False)
    release_mem(fig)


def task1_struct_params_vs_dmin(task, fixP, fixT, fixV):

    # morph_file = input_dir+"/feature/task3/bulk_label_"+fixP+fixT+fixV+"_morphology.txt"
    # morph_val = np.loadtxt(morph_file, dtype=str).ravel()
    # bkg_idx = np.where(morph_val=="bkg")[0]

    # # get dmin
    dmin_value = get_dmin(fixP, fixT, fixV)
    xlabel = "dmin"
    feature = "Pt-density"

    ylims_range = dict({"Pt-density":(-0.0001, 0.0028), 
        "Pt-valence":(0.0, 1.25), "Pt-O":(-0.01, 0.45), "Pt-Pt":(0.0, 12.2)})
    for feature in ["Pt-density", "Pt-valence", "Pt-O", "Pt-Pt"]:
        xlim=[-2, 50]
        ylim=ylims_range[feature]


        ft_file, save_at, save_at_sb = task1_get_ftfile_saveat(task, fixP, fixT, fixV, feature)
        ft_val = np.loadtxt(ft_file).ravel()
        # ft_val =ft_val*100

        dmin_value_copy = copy.copy(dmin_value)
        bkg_idx = np.where(dmin_value==-50)[0]

        df_Ptdens = remove_nan(ignore_first=bkg_idx,
            matrix1=dmin_value_copy,matrix2=ft_val,lbl1=xlabel,lbl2=feature)

        # if fixT != "ADT5k":
        #     y = y * 10000
        # joint_plot_2(x=x, y=y, xlabel=xlabel, ylabel=feature, 
        #     xlim=xlim, ylim=[-2, 50], 
        #     title=save_at.replace(result_dir, ""), save_at=save_at)
        # joint_plot(x=x, y=y, xlabel=xlabel, ylabel=feature, 
        #     xlim=xlim, ylim=ylim, 
        #     title=save_at.replace(result_dir, ""),
        #     save_at=save_at.replace(".pdf", "_2.pdf"))
        plot_joinmap(df_Ptdens, selected_inst=None, xlbl=xlabel, ylbl=feature, 
                    xlbl_2fig="Distance to surface", ylbl_2fig=feature, color="blue",
                    save_at=save_at, 
                    is_gmm_xhist=False, is_gmm_yhist=False, 
                    means=None, weight=None, cov_matrix=None, 
                    n_components=None, xlim=xlim, ylim=ylim,
                    main_ax_type="kde")




def task2_deltaPT_vs_dmin(fixT, fixP, fixV, diff_state, task):
    final_state, init_state = diff_state 

    # # # only for dt
    fix_val = "{0}{1}".format(fixP, fixV)

    # # dminCCM-NafionADT15k04V_morphology.txt
    # # dminCCM-NafionFresh04V_morphology.txt

    # # # get dmin
    # dmin_file = "{0}/task4/dmin{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
    # dmin_value = np.loadtxt(dmin_file)
    # xlim=[-2, 90]

    # # # get Pt-density value

    # consider_Ft = "Pt-density"
    # prefix_input = fix_val + consider_Ft
    # diff_Ptdens = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
    #                 "diff_t", prefix_input, 
    #                 final_state, init_state)
    # diff_Ptdens_val, is_diff_Ptdens_pos = pos_neg_lbl_cvt(inputfile=diff_Ptdens)

    # diff_Ptdens_val_fix = 0-diff_Ptdens_val
    # # # 2d image
    # save_at = "{0}/new_request/{1}/img_{2}_{3}___{4}.pdf".format(result_dir,
    #     task, fix_val, final_state,  init_state) 
    # plot_density(values=diff_Ptdens_val_fix, save_at=save_at,  
    #     # cmap_name="Oranges", vmin=None, vmax=None, is_save2input=False
    #     cmap_name="bwr", vmin=-0.004, vmax=0.004
    #     )
# plot_density(matrix_transform_to_plot, path + output_image + "bulk_label_"+p + str(file_init.replace(".txt","")),str(file_init.replace(".txt","")) ,  )
    # morph_file = input_dir+"/feature/task3/bulk_label_"+fixP+fixT+fixV+"_morphology.txt"
    # morph_val = np.loadtxt(morph_file, dtype=str).ravel()
    # bkg_idx = np.where(morph_val=="bkg")[0]


    save_at = "{0}/new_request_3/task2_deltaPt_at{5}/dmin_Ptdens_{1}_{2}_{3}___{4}.pdf".format(result_dir,
        task, fix_val, final_state,  init_state, fixT) 


    dmin_value = get_dmin(fixP, fixT, fixV)
    dmin_value_copy = copy.copy(dmin_value)

    diff_Ptdens = "{0}/feature/task1/diff_t/{1}{2}Pt-density_{3}____{4}.txt".format(input_dir, fixP, fixV, final_state, init_state)
    diff_Ptdens_val = np.loadtxt(diff_Ptdens).ravel()

    xlabel = "dmin"
    ylabel = "delta_Pt-dens"

    bkg_idx = np.where(dmin_value==-50)[0]
    df_Ptdens = remove_nan(ignore_first=bkg_idx,
            matrix1=dmin_value_copy,matrix2=diff_Ptdens_val,lbl1=xlabel,lbl2=ylabel)

    xlim=[-2, 50]
    ylim=(-max_Ptdens, max_Ptdens)
    plot_joinmap(df_Ptdens, selected_inst=None, xlbl=xlabel, ylbl=ylabel, 
            xlbl_2fig="Distance to surface at\n"+fixP+fixT+fixV, ylbl_2fig=ylabel, color="blue",
            save_at=save_at, 
            is_gmm_xhist=False, is_gmm_yhist=False, 
            means=None, weight=None, cov_matrix=None, 
            n_components=None, xlim=xlim, ylim=ylim,
            main_ax_type="kde")

 
def task3_reaction_mode(fixT, fixP, fixV, diff_state):
    # ADT5k1VPt-density_CCM-Nafion____CCMcenter.pdf
    final_state, init_state = diff_state # # here is diff voltage

    task = "diff_t"
    fix_val = "{0}{1}".format(fixP, fixV)
    
    consider_Ft = "Pt-O"
    prefix_input = fix_val + consider_Ft
    diff_PtO = "{0}/feature/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
                    task, prefix_input, 
                    final_state, init_state)
    diff_PtO_val, is_diff_PtO_pos = pos_neg_lbl_cvt(inputfile=diff_PtO)


    consider_Ft = "Pt-valence"
    prefix_input = fix_val + consider_Ft
    diff_PtVal = "{0}/feature/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
                    task, prefix_input, 
                    final_state, init_state)
    diff_PtVal_val, is_diff_PtVal_pos = pos_neg_lbl_cvt(inputfile=diff_PtVal)



    consider_Ft = "Pt-Pt"
    prefix_input = fix_val + consider_Ft
    diff_PtPt = "{0}/feature/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
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
    dmin_file = "{0}/feature/task4/dmin{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
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
    task = "task2_deltaPt" # task1_dmin_corr, task2_deltaPt, task3

    if task == "task1_dmin_corr":
        fixV = "04V"
        for fixP in Positions:
            for fixT in ["Fresh", "ADT15k"]: # Times
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
