import random, pickle
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import rc
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import matplotlib.patches as mpatches

from plot import joint_plot_1, ax_setting, makedirs, release_mem, plot_density

colors = ["Blues", "Greens", "Oranges", "Reds"]
axis_font = {'fontname': 'serif', 'size': 16, 'labelpad': 8}

y_size = 512 
x_size = 512 


def plot_kde(X_plot, n_components, labels, ax, 
    xlbl, ylbl, plot_type='kde', margin=False,):
    sns.set_style('ticks')
    label_patches = []
    for k in range(n_components):
        g_inst = np.where(labels == k)
        x = X_plot[g_inst, 0]
        y = X_plot[g_inst, 1]

        x1 = pd.Series(x[0], name=xlbl)
        x2 = pd.Series(y[0], name=ylbl)

        # if k == 0:
        #     levels = [9, 12, 15]

        if plot_type == 'kde':
            ax = sns.kdeplot(x1, x2,
                             #joint_kws={"colors": "black", "cmap": None, "linewidths": 0.5},
                             cmap=colors[k],
                             shade=False, shade_lowest=False,
                             # n_levels=levels,
                             fontsize=10,
                             ax=ax,
                             vertical=True)
            label_patch = mpatches.Patch(
                    color=sns.color_palette(colors[k])[2],
                    label="Group {0}".format(k + 1))
            label_patches.append(label_patch)
        elif plot_type == 'hex':
            plt.hexbin(x1, x2, gridsize=20, cmap=colors[k-1], marginals=True, mincnt=2)
        # plt.scatter(x, y,s=20, alpha=0.2, c=color_single[k],
        #             edgecolors='none',
        #             label='Group {0}'.format(k+1))
        # plt.tick_params(labelleft=False, labelbottom=False)
    
    # ax.set_xlabel(xlbl, fontsize=10)
    # ax.set_tlabel(ylbl, fontsize=10)
    # plt.xlabel("")
    # plt.ylabel("")
    # if plot_type == 'kde':
    #     ax.tick_params(axis='both', which='major', labelsize=24)
    # leg = ax.axis.get_legend()
    # new_labels = ["Group {0}".format(k + 1) for k in range(n_components)]
    # for t, l in zip(leg.texts, new_labels): t.set_text(l)
    ax.legend(handles=label_patches, loc='upper right')
    return ax


def plot_kde_subaxis(X_plot, n_components, labels, ax, 
        lbl, plot_type='kde', margin=False,):
    sns.set_style('ticks')
    xgroup = []
    for k in range(n_components):
        g_inst = np.where(labels == k)
        x = X_plot[g_inst]
        xgroup.append(x)
        # y = X_plot[g_inst, 1]

        # x1 = pd.Series(x[0], name=lbl)
        # x2 = pd.Series(y[0], name=ylbl)


        if n_components == 2:
            colors = ["blue", "red"]
        elif n_components == 3:
            colors = ["blue", "green", "red"]
        elif n_components == 4:
            colors = ["purple", "blue", "orange", "red"]
        if plot_type == 'kde':
            ax = sns.distplot(x, color=colors[k],
                             ax=ax, norm_hist=False,
                             vertical=True
                             ) #distplot


    #         #
    # ax.hist(xgroup, 50, histtype='stepfilled', stacked=False, color=colors, alpha=0.5)
    # ax.hist(X_plot, 50, histtype='step', stacked=True, color='black', alpha=1.0)

    # ax = sns.distplot(X_plot, color='grey',
    #       ax=ax, norm_hist=True
    #       )
    # ax.legend(handles=label_patches, loc='upper right')
    return ax


def plot_sub_gmm(means, weights, covars, data, ax, n_components, 
            xlim=None, ylim=None, orientation='vertical'):
    import scipy.stats as stats

    n, bins, patches = ax.hist(data, 100, normed=True, 
        facecolor='grey', alpha=0.3, orientation=orientation)



    if n_components == 2:
        colors = ["blue", "red"]
    elif n_components == 3:
        colors = ["blue", "green", "red"]
    elif n_components == 4:
        colors = ["purple", "blue", "orange", "red"]
    x = np.linspace(np.min(data), np.max(data), 100)[:, np.newaxis]
    for i in range(n_components):
        this_y = weights[i] * stats.norm.pdf(x,means[i],np.sqrt(covars[i]))
        if orientation == 'vertical':
            ax.plot(x, this_y, c=colors[i])
        else:
            ax.plot(this_y, x, c=colors[i])


def get_best_gmm(X_matrix, n_components, score_df,  n_sampling=20, means_init=None):
    n_points = len(X_matrix)
    for i in range(n_sampling):
        gmm = GaussianMixture(n_components=n_components,
            # reg_covar=0.0000001
            # covariance_type='full',
                            means_init=means_init,
        # #weights_init = [0.1, 0.33, 0.26, 0.1]
        # init_params='random'
                              )
        sample = random.sample(range(n_points), int(0.8*n_points))
        X_rand = X_matrix[sample]
        # X_rand = X_matrix

        # np.random.shuffle(X_matrix)
        gmm.fit(X=X_rand)
        this_AIC = gmm.aic(X=X_rand)
        this_BIC = gmm.bic(X=X_rand)
        score_df.loc[i, "AIC"] = this_AIC
        score_df.loc[i, "BIC"] = this_BIC

        if i == 0:
            best_AIC = this_AIC
            best_gmm = gmm
        else:
            if this_AIC < best_AIC:
                best_AIC = this_AIC
                best_gmm = gmm

    return best_gmm, score_df


def re_arrange_gmm_lbl(means, labels, sort_ax):
    means = np.array(means)
    labels = np.array(labels)

    # sorting gmm components by means value along sort_ax
    means_argsort = means[:, sort_ax].argsort()
    print (means, means_argsort, means[means_argsort])
    # create translation dict: store original label at k, new label after sorting at v
    trans_dict = dict({k:v for k, v in zip(means_argsort, range(len(means)))})
    print (trans_dict)
    # translate
    new_labels = np.vectorize(trans_dict.get)(labels)

    return new_labels, means_argsort


def save_output(filename, means, cov_matrix, weight, axis_sort, sort_index):
    v, w = np.linalg.eigh(cov_matrix)

    with open(filename,"w+") as center_file:
        center_file.write("Centers:\n {0}\n=======\n".format(means))

        center_file.write("Covariance matrix:\n {0}\n=======\n".format(cov_matrix))
        center_file.write("Eigen value:\n {0} \n=======\n".format(v))
        center_file.write("Eigen vector:\n {0} \n=======\n".format(w)), 
        center_file.write("Weight:\n {0} \n=======\n".format(weight))
        center_file.write("Axis sort:\n {0} \n=======\n".format(axis_sort))
        center_file.write("Sort index:\n {0} \n=======\n".format(sort_index))


def get_df_xylbl_XGmm(inputdf, gmm_var, xlbl_plot, ylbl_plot, bulk_inst):
    is_random_subset = True

    if xlbl_plot is None and ylbl_plot is None:
        xlbl = gmm_var[0]
        ylbl = gmm_var[-1]
    else:
        xlbl = xlbl_plot
        ylbl = ylbl_plot 


    if bulk_inst is not None:
        df = inputdf.loc[bulk_inst, [xlbl, ylbl]].dropna()
    else:
        df = inputdf[[xlbl, ylbl]].dropna()
        # df[gmm_var] = df[gmm_var]*100

    indexes = df.index
    n_inst = len(indexes)

    if is_random_subset:
        # random subset selection
        selected_inst =  indexes[np.random.randint(0, n_inst, size=100000)]
    else:
        selected_inst = indexes

    if len(gmm_var) == 1:
        X_matrix_search = df.loc[selected_inst, gmm_var].values.reshape(-1, 1) 
    else:
        X_matrix_search = df.loc[selected_inst, gmm_var].values

    return xlbl, ylbl, df, X_matrix_search, selected_inst


def plot_joinmap(df, selected_inst, xlbl, ylbl, new_labels, gmm_var, 
                xlbl_2fig=None, ylbl_2fig=None, 
                savedir=None, 
                is_gmm_xhist=True, is_gmm_yhist=True, 
                means=None, weight=None, cov_matrix=None, 
                
                n_components=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    main_ax = fig.add_subplot(grid[1:, :-1])
    y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
    

    sns.set_style('ticks')


    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # plot kde contour of each components
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    main_ax = plot_kde(X_plot=df.loc[selected_inst, [xlbl, ylbl]].values ,
                n_components=n_components,
                labels=new_labels, plot_type='kde', ax=main_ax, xlbl=xlbl, ylbl=ylbl)
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
        if len(gmm_var) == 1:
            plot_sub_gmm(means=means, weights=weight, covars=cov_matrix,
                data=df.loc[selected_inst, xlbl].values, ax=x_hist, n_components=n_components,
                xlim=xlim, ylim=ylim)
        else:
            plot_sub_gmm(means=means[0], weights=weight, covars=cov_matrix[0].diagonal(),
                data=df.loc[selected_inst, xlbl].values, ax=x_hist, n_components=n_components,
                xlim=xlim, ylim=ylim)
    else:
        sns.distplot(df.loc[selected_inst, xlbl].values,  color='grey', ax=x_hist, norm_hist=False)
    print ("Done plot is_gmm_xhist")

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # plot gmm_yhist
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    if is_gmm_yhist:

        plot_sub_gmm(means=means, weights=weight, covars=cov_matrix,
                  data=df.loc[selected_inst, ylbl].values, ax=y_hist, n_components=n_components,
                  orientation='horizontal')
    else:
        sns.distplot(df.loc[selected_inst, ylbl].values,  color='grey', ax=y_hist, 
            vertical=True, norm_hist=False)

    main_ax.tick_params(axis='both', labelsize=10)
    if xlim is not None:
        main_ax.set_xlim(xlim)
        x_hist.set_xlim(xlim)

    if ylim is not None:
        main_ax.set_ylim(ylim)
        y_hist.set_ylim(ylim)

    # plt.legend()
    plt.setp(x_hist.get_xticklabels(), visible=False)
    plt.setp(y_hist.get_yticklabels(), visible=False)
    plt.tight_layout(pad=1.1)

    savefile = "{0}/gmm.pdf".format(savedir)
    plt.savefig(savefile, transparent=False)
    release_mem(fig)


def GaussianMixtureModel(inputdf, gmm_var, savedir, 
                    cmap_name, bulk_inst, is_2densitymap=False, 
                    n_components=3, 
                    sort_ax_gmm=1, 
                    xlbl_plot=None, ylbl_plot=None, 
                    xlbl_2fig=None, ylbl_2fig=None,
                    is_gmm_xhist=True, is_gmm_yhist=False,
                    xlim=None, ylim=None, means_init=None,
                    is_save_gmm = False, # for the case we need to save best_gmm file, most use for fit all 
                    is_exist_gmm = False, # for the case gmm files already exist, no need to fit again
                    best_gmm_file=None):
    n_sampling = 20
    is_need_add_zero = False
    """
        inputdf: index: pixel index of each image
        gmm_var: [feature1, feature2]
        bulk_inst: pixel with label "bulk"
        is_save_gmm: for the case we need to save best_gmm file, most use for fit all
        is_exist_gmm: for the case gmm files already exist, no need to fit again
    """

    xlbl, ylbl, df, X_matrix_search, selected_inst = get_df_xylbl_XGmm(inputdf=inputdf, gmm_var=gmm_var, 
        xlbl_plot=xlbl_plot, ylbl_plot=ylbl_plot, bulk_inst=bulk_inst)
    

    print (np.min(X_matrix_search), np.max(X_matrix_search))
    #X_matrix_search = df.loc[:, gmm_var]
    score_df = pd.DataFrame(index=range(n_sampling), columns=["AIC", "BIC"])

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # add zero points
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    if is_need_add_zero:
        zeros = [[0.0]] * 20000
        X_matrix_search_tmp = np.concatenate((X_matrix_search, zeros))
        print (X_matrix_search_tmp)
    else:
        X_matrix_search_tmp = X_matrix_search

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # get best_gmm model, either load or train
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    if is_exist_gmm:
        # if exist best_gmm, then load best_gmm in "best_gmm_file" (same to gmm_file2save)
        # best_gmm_file = "{0}/best_gmm.pickle".format(savedir)
        best_gmm = pickle.load(open(best_gmm_file, 'rb'))

    else:
        # if no exist best_gmm model, then find it
        best_gmm, score_df = get_best_gmm(X_matrix=X_matrix_search_tmp,  # .reshape(-1, 1) 
            n_components=n_components, n_sampling=n_sampling, score_df=score_df,
            means_init=means_init)
        
        # X_matrix = np.array(df[gmm_var])
        # best_gmm.fit(X=X_matrix_search)
        makedirs("{0}/GMM_score.csv".format(savedir))
        score_df.to_csv("{0}/GMM_score.csv".format(savedir))

        if is_save_gmm:
            gmm_file2save = "{0}/best_gmm.pickle".format(savedir)
            makedirs(gmm_file2save)
            pickle.dump(best_gmm, open(gmm_file2save, 'wb'))

        # # # # # # # # # # # # # # # # # # # # # # #
        #
        # save pre transform model to file
        #
        # # # # # # # # # # # # # # # # # # # # # # #
    means = best_gmm.means_
    cov_matrix = best_gmm.covariances_
    weight = best_gmm.weights_
    txt2save = "{0}/gmm_center_non_transform.txt".format(savedir)
    makedirs(txt2save)
    save_output(filename=txt2save, means=means, cov_matrix=cov_matrix, 
                weight=weight,
                axis_sort=None, sort_index=None)
        

    # print (gmm_var)
    # labels = best_gmm.predict(X=df.loc[selected_inst, gmm_var].values.reshape(-1, 1))
    labels = best_gmm.predict(X=X_matrix_search)

    print (best_gmm)
    print (set(labels))
    print (means, cov_matrix)
    new_labels, means_argsort = re_arrange_gmm_lbl(means=means, 
        labels=labels, sort_ax=sort_ax_gmm)

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # save after transform model to file
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    means = means[means_argsort]
    cov_matrix = cov_matrix[means_argsort]
    weight = weight[means_argsort]
    txt2save = "{0}/gmm_center_transform.txt".format(savedir)
    makedirs(txt2save)
    save_output(filename=txt2save, means=means, cov_matrix=cov_matrix, weight=weight,
         axis_sort=sort_ax_gmm, sort_index=means_argsort)


    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # create a main axis to plot, x_axis and y_axis
    # 
    # # # # # # # # # # # # # # # # # # # # # # #
    if is_2densitymap:
        plot_joinmap(df=df, selected_inst=selected_inst, 
                xlbl=xlbl, ylbl=ylbl, xlbl_2fig=xlbl_2fig, ylbl_2fig=ylbl_2fig, 
                new_labels=new_labels, gmm_var=gmm_var,
                savedir=savedir, 
                is_gmm_xhist=is_gmm_xhist, is_gmm_yhist=is_gmm_yhist, 
                means=means, weight=weight, cov_matrix=cov_matrix, 
                n_components=n_components, xlim=xlim, ylim=ylim)


    # # # # # # # # # # # # # # # # # # # # # # #
    #
    # plot density map
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    if is_2densitymap:
        inputdf["gmm_lbl"] = np.nan
        inputdf.loc[selected_inst, "gmm_lbl"] = new_labels + 1
        gmm_lbl_df = pd.DataFrame(inputdf["gmm_lbl"].values.reshape(x_size, y_size).T)
        gmm_lbl_df.to_csv("{0}/gmm_label.csv".format(savedir))
        save_at =  "{0}/gmm_all".format(savedir)
        plot_density(values=gmm_lbl_df.values, save_at=save_at, 
            vmin=1.0, vmax=n_components,
            cmap_name=cmap_name)
