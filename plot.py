import numpy as np
import matplotlib.pyplot as plt
import time, gc, os
import pandas as pd

import sys 
sys.path.append("..")
try:
	import seaborn as sns
except:
	pass
import itertools
from scipy import stats
from sklearn import mixture
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity


axis_font = {'fontname': 'serif', 'size': 16, 'labelpad': 8}
title_font = {'fontname': 'serif', 'size': 12}
size_point = 3
alpha_point = 0.3
n_neighbor = 3

sns.palplot("dark")
sns.set_style('ticks')

def release_mem(fig):
	fig.clf()
	plt.close()
	gc.collect()

def ax_setting(ax=None):
	plt.style.use('default')
	plt.tick_params(axis='x', which='major', labelsize=13)
	plt.tick_params(axis='y', which='major', labelsize=13)
	if ax is not None:
		ax.tick_params(axis="x", direction='in', width=2, length=20)
		ax.tick_params(axis="y", direction='in', width=2, length=20)
		ax2 = ax.twinx()
		ax3 = ax.twiny()
		ax2.tick_params(axis="y", direction="in", width=2, length=20, labelcolor="white")
		ax3.tick_params(axis="x", direction='in', width=2, length=20, labelcolor="white")


def makedirs(file):
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))


def ax_setting_3d(ax):
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)
	ax.xaxis._axinfo["grid"]['color'] =  "w"
	ax.yaxis._axinfo["grid"]['color'] =  "w"
	ax.zaxis._axinfo["grid"]['color'] =  "w"

	# ax.set_xticks([])
	# ax.set_zticks([])

	ax.tick_params(axis='x', which='major', labelsize=20)
	ax.tick_params(axis='y', which='major', labelsize=20)
	ax.tick_params(axis='z', which='major', labelsize=20, pad=30)

	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False


	ax.xaxis.pane.set_edgecolor('w')
	ax.yaxis.pane.set_edgecolor('w')
	ax.zaxis.pane.set_edgecolor('w')

	ax.zaxis.set_rotate_label(False)

	ax.view_init(elev=30.)
	# ax.view_init(45, 120) # good for Tc
	plt.tight_layout(pad=1.1)


def scatter_plot(x, y, xvline=None, yhline=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	linestyle='-.', marker='o'):
	fig = plt.figure(figsize=(8, 8))

	if 'scatter' in mode:
		plt.scatter(x, y, s=100, alpha=0.8, 
		marker=marker, c=color, edgecolor="white") # brown

	if 'line' in mode:
		plt.plot(x, y,  marker=marker, linestyle=linestyle, color=color,
		 alpha=1.0, label=lbl, markersize=10, mfc='none')

	if xvline is not None:
		plt.axvline(x=xvline, linestyle='-.', color='black')
	if yhline is not None:
		plt.axhline(y=yhline, linestyle='-.', color='black')

	if name is not None:
		for i in range(len(x)):
			# only for lattice_constant problem, 1_Ag-H, 10_Ag-He
			# if tmp_check_name(name=name[i]):
			   # reduce_name = str(name[i]).split('_')[1]
			   # plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)
			plt.annotate(name[i], xy=(x[i], y[i]), size=6)
		

	plt.ylabel(y_label, **axis_font)
	plt.xlabel(x_label, **axis_font)
	ax_setting()



	# plt.grid(linestyle='--', color="gray", alpha=0.8)
	plt.legend(prop={'size': 16})
	makedirs(save_file)
	plt.savefig(save_file)
	release_mem(fig=fig)



def scatter_plot_4(x, y, color_array=None, xvlines=None, yhlines=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	s=100, alphas=0.8, title=None,
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	preset_ax=None, linestyle='-.', marker='o'):


	fig = plt.figure(figsize=(8, 8), linewidth=1.0)
	grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
	main_ax = fig.add_subplot(grid[1:, :-1])
	y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
	x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
	
	sns.set_style(style='white') 

	# main_ax.legend(lbl, 
	#   loc='lower left', fontsize=18,
	#   bbox_to_anchor=(1.05, 1.05, ),  borderaxespad=0)
	plt.title(title)


	# elif isinstance(marker, list):
	main_ax = sns.kdeplot(x, y,
			 # joint_kws={"colors": "black", "cmap": None, "linewidths": 3.0},
			 cmap='Oranges',
			 shade=True, shade_lowest=False,
			 fontsize=10, ax=main_ax, linewidths=1,
			 # vertical=True
			 )
	if color_array is None:
	    main_ax.scatter(x, y, s=s, alpha=0.8, marker=marker, c=color, 
	        edgecolor="black")
	# for _m, _c, _x, _y, _a in zip(marker, color_array, x, y, alphas):
	# 	main_ax.scatter(_x, _y, s=s, marker=_m, c=_c, alpha=_a, edgecolor="black")

	

	# for xvline in xvlines:
	#   main_ax.axvline(x=xvline, linestyle='-.', color='black')
	# for yhline in yhlines:
	#   main_ax.axhline(y=yhline, linestyle='-.', color='black')

	main_ax.set_xlabel(x_label, **axis_font)
	main_ax.set_ylabel(y_label, **axis_font)
	if name is not None:
		for i in range(len(x)):
			# only for lattice_constant problem, 1_Ag-H, 10_Ag-He
			# if tmp_check_name(name=name[i]):
			   # reduce_name = str(name[i]).split('_')[1]
			   # plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)
			main_ax.annotate(name[i], xy=(x[i], y[i]), size=size_text)

	# x_hist.hist(x, c='orange', linewidth=1)
	# y_hist.hist(y, c='orange', linewidth=1)
	# red_idx = np.where((np.array(color)=="red"))[0]


	# # x-axis histogram
	sns.distplot(x, bins=100, ax=x_hist, hist=False,
		kde_kws={"color": "grey", "lw": 1},
		# shade=True,
		# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "orange"},
		vertical=False, norm_hist=True)
	l1 = x_hist.lines[0]
	x1 = l1.get_xydata()[:,0]
	y1 = l1.get_xydata()[:,1]
	x_hist.fill_between(x1, y1, color="orange", alpha=0.3)

	# sns.distplot(x[red_idx], bins=100, ax=x_hist, hist=False,
	# 	kde_kws={"color": "blue", "lw": 1},
	# 	# shade=True,
	# 	# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
	# 	vertical=False, norm_hist=True)
	# l1 = x_hist.lines[0]
	# x1 = l1.get_xydata()[:,0]
	# y1 = l1.get_xydata()[:,1]
	# x_hist.fill_between(x1, y1, color="blue", alpha=0.3)

	# # y-axis histogram
	sns.distplot(y, bins=100, ax=y_hist, hist=False,
		kde_kws={"color": "grey", "lw": 1},
		# shade=True,
		# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "orange"},
		vertical=True, norm_hist=True)
	l1 = y_hist.lines[0]
	x1 = l1.get_xydata()[:,0]
	y1 = l1.get_xydata()[:,1]
	y_hist.fill_between(x1, y1, color="orange", alpha=0.3)


	# sns.distplot(y[red_idx], bins=100, ax=y_hist, hist=False,
	# 	kde_kws={"color": "blue", "lw": 1},
	# 	# shade=True,
	# 	# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
	# 	vertical=True, norm_hist=True)
	# l1 = y_hist.lines[0]
	# x1 = l1.get_xydata()[:,0]
	# y1 = l1.get_xydata()[:,1]
	# y_hist.fill_between(x1, y1, color="blue", alpha=0.3)



	plt.setp(x_hist.get_xticklabels(), visible=False)
	plt.setp(y_hist.get_yticklabels(), visible=False)
	plt.tight_layout(pad=1.1)

	makedirs(save_file)
	plt.savefig(save_file, transparent=False)
	print ("Save at: ", save_file)
	release_mem(fig=fig)



def plot_density(values, save_at,  cmap_name, 
	title, vmin, vmax, is_save2input=None,
	is_lbl=False, set_labels=None):
	# input: matrix [n_rows, n_cols] of any value
	# output: figure
	fig = plt.figure(figsize=(10, 10), dpi=300)

	# if vmin is None and vmax is None:
	# 	max_abs = 0.95 * max((abs(np.min(values)), abs(np.max(values))))
	# 	vmin = -max_abs
	# 	vmax = max_abs

	# m = np.ma.masked_where(np.isnan(values),values)
	# plt.pcolor(m)
	# img = plt.imread(values, format='raw')

	cmap = plt.get_cmap(cmap_name)
	cmap.set_bad('white')
	# img = cmap(values)
	# print (img)
	print ("here", vmin, vmax)

	if is_lbl:
		for lbl in set_labels:
			first_idx = np.where(values==lbl)
			if len(first_idx[0]) != 0:
				yt, xt =  first_idx[0][0], first_idx[1][0]
				plt.text(xt, yt, lbl)
	plt.imshow(values, cmap=cmap, 
		interpolation='none', vmin=vmin, vmax=vmax)

	# plt.colorbar()
	# plt.xlabel('x', **axis_font)
	# plt.ylabel('y', **axis_font)

	# title = save_at[save_at.find('result/') + len('result/'): ]
	# plt.title(title, **title_font)
	ax = plt.gca()
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)

	makedirs(save_at)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	plt.savefig(save_at, transparent=False)
	print ("Save file at:", save_at)

	# # save to input
	if is_save2input is not None:
		# save_txt = save_at.replace("result", "input").replace(".pdf", ".txt")
		makedirs(is_save2input)
		print("Redox save at:", is_save2input)
		np.savetxt(is_save2input, values)

	release_mem(fig)


def plot_hist(x, save_at=None, label=None, nbins=50):

	if save_at is not None:
		fig = plt.figure(figsize=(16, 16))

	# hist, bins = np.histogram(x, bins=300, normed=True)
	# xs = (bins[:-1] + bins[1:])/2

	# plt.bar(xs, hist,  alpha=1.0)
	# y_plot = hist
	y_plot, x_plot, patches = plt.hist(x, bins=nbins, histtype='stepfilled', # step, stepfilled, 'bar', 'barstacked'
										density=True, label=label, log=False,  
										color='black', #edgecolor='none',
										alpha=1.0, linewidth=2)

	# X_plot = np.linspace(np.min(x), np.max(x), 1000)[:, np.newaxis]
	# kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(x.reshape(-1, 1))
	# log_dens = kde.score_samples(X_plot)
	# plt.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
	# plt.text(-3.5, 0.31, "Gaussian Kernel Density")

	#plt.xticks(np.arange(x_min, x_max, (x_max - x_min) / 30), rotation='vertical', size=6)
	# plt.ylim([1, np.max(y_plot)*1.01])
	plt.legend()
	plt.ylabel('Probability density', **axis_font)
	plt.xlabel("Value", **axis_font)

	ax_setting()

	if save_at is not None:

		if not os.path.isdir(os.path.dirname(save_at)):
			os.makedirs(os.path.dirname(save_at))
		plt.savefig(save_at)
		print ("Save file at:", "{0}".format(save_at))
		release_mem(fig)

	# return y_plot


def plt_hist_gmm(X_plot, save_fig_file, label, is_kde=False,  is_gmm=True, n_components_gmm=3, save_gmm_file=None,
				means_init=None, weighs_init=None):
	from opt_GMM import opt_GMM
	# X_plot = X_plot[X_plot < 600]
	nbins = 200
	x_lb = np.min(X_plot)
	x_ub = np.max(X_plot)
	bandwidth = (x_ub - x_lb) / nbins

	fig = plt.figure(figsize=(8, 8))

	# plot hist of ensembling
	if True:
		plot_hist(x=X_plot, label=label, nbins=nbins)

	# plot kde
	X_kde = np.linspace(x_lb, x_ub, nbins)[:, np.newaxis]

	# plot KernelDensity
	if is_kde: # set True
		kde = KernelDensity(kernel="gaussian", 
			bandwidth=bandwidth).fit(X_plot.reshape(-1, 1))
		print (np.min(X_plot), np.max(X_plot))
		log_dens = kde.score_samples(X_kde)
		plt.plot(X_kde, np.exp(log_dens), '-', label="KDE", linewidth=3, c="black")


	# plot GMM
	if is_gmm:
		ax_setting()
		this_mean_init = None
		this_weigh_init = None
		try:
			this_mean_init = means_init[test_point]
			this_weigh_init = weighs_init[test_point]
		except:
			pass

		# optimize
		best_gmm = opt_GMM(X=X_plot.reshape(-1, 1), n_sampling=200, 
			n_components=n_components_gmm, means_init=this_mean_init)
		best_gmm.fit(X_plot.reshape(-1, 1))

		# save
		if save_gmm_file is not None:
			pickle.dump(best_gmm, open(save_gmm_file, 'wb'))


		weights = best_gmm.weights_
		means = best_gmm.means_
		covars = best_gmm.covariances_

		print ("Best_GMM: ", weights, means, covars,)
		
		# plot gaussian components
		colors = ["red", "blue", "green", "black", "orange", 
					"brown", "purple", "cyan", "teal", "wheat", 'mediumslateblue', 
					'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',  
					'mintcream', 'mistyrose', 'moccasin', 'slateblue', 'slategray', 
					'slategrey', 'snow', 'springgreen', 'steelblue', 
					'tan', 'teal', 'thistle', 'tomato',]
		for ii, (w, m, c, cl) in enumerate(zip(weights, means, covars, colors)):
			# print (test_point, w, m, c)
			this_X_plt = X_kde
			plt.fill(this_X_plt, w*stats.norm.pdf(this_X_plt, m, np.sqrt(c)), 
				label="Mean {0}: {1}".format(int(ii +1), round(m[0],2)), c=cl, alpha=0.8)
		

	
	plt.legend(prop={'size': 16})
	plt.savefig(save_fig_file)
	release_mem(fig)


def joint_plot_1(x, y, xlabel, ylabel, xlim, ylim, title, save_at):
	import scipy.stats as st

	fig = plt.figure(figsize=(8, 8))
	# sns.set_style('ticks')
	sns.plotting_context(font_scale=1.5)
	this_df = pd.DataFrame()
	
	this_df[xlabel] = x
	this_df[ylabel] = y
	# try:
		# g = sns.jointplot(this_df[xlabel], this_df[ylabel],
		# 			kind="kde", shade=True, 
		# 			color="green",
		# 			xlim=xlim, ylim=ylim)

	# plt.hist2d(x,y,100,cmap='jet')
	xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values)
	f = np.reshape(kernel(positions).T, xx.shape)

	ax = fig.gca()
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	# Contourf plot
	cfset = ax.contourf(xx, yy, f, cmap='Oranges')
	## Or kernel density estimate plot instead of the contourf plot
	#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
	# Contour plot
	cset = ax.contour(xx, yy, f, colors='k')
	# Label plot
	ax.clabel(cset, inline=1, fontsize=10)
	# ax.set_xlabel(xlabel)
	# ax.set_ylabel(ylabel)
	# except Exception as e:
	# 	g = sns.jointplot(this_df[xlabel], this_df[ylabel],
	# 				kind="hex",  # shade=True, 
	# 				color="green",
	# 				xlim=xlim, ylim=ylim)

	
	# g.ax_marg_x.set_axis_off()
	# g.ax_marg_y.set_axis_off()
	# ax.set_xlabel(xlabel, **axis_font)
	# ax.set_ylabel(ylabel, **axis_font)

	# ax.set_xlim(xlim)
	# ax.set_ylim(ylim)
	# ax.spines['right'].set_visible(False)
	# ax.spines['top'].set_visible(False)
	plt.xlabel(r'%s' %xlabel, **axis_font)
	plt.ylabel(r'%s' %ylabel, **axis_font)
	plt.title(title, **title_font)

	# plt.set_tlabel('sigma', **axis_font)
	# ax_setting()
	makedirs(save_at)
	plt.savefig(save_at)
	plt.tight_layout(pad=1.1)

	print ("Save file at:", "{0}".format(save_at))
	release_mem(fig)


def joint_plot(x, y, xlabel, ylabel, xlim, ylim, title, save_at, is_show=False):
	fig = plt.figure(figsize=(12, 12))
	# sns.set_style('ticks')
	sns.plotting_context(font_scale=1.5)
	this_df = pd.DataFrame()
	
	this_df[xlabel] = x
	this_df[ylabel] = y

	ax = sns.jointplot(this_df[xlabel], this_df[ylabel],
					kind="kde",  shade=True, # hex
					# xlim=xlim, ylim=ylim,
					color='orange',).set_axis_labels(xlabel, ylabel)

	# ax = ax.plot_joint(plt.scatter,
	# 			  color="grey", s=2, edgecolor=None)
	# ax.scatter(x, y, s=30, alpha=0.5, c='red')
	# ax.spines['right'].set_visible(False)
	# ax.spines['top'].set_visible(False)
	# plt.xlabel(r'%s' %xlabel, **axis_font)
	# plt.ylabel(r'%s' %ylabel, **axis_font)
	# ax.title(title, title_font)

	# plt.set_tlabel('sigma', **axis_font)
	# ax_setting(ax)
	# plt.yticks(np.arange(0, ylim[1], 10)) # fontsize=12
	plt.subplots_adjust(top=1.1)
	ax.fig.suptitle(title) 
	plt.yticks(np.arange(0, ylim[1], 10)) # # Ptdens: 10, valence: 0.5, Pt-Pt: 2

	# ax.setxlim(xlim)
	# # ax.setylim(ylim)
	# ax.set_xticks(np.arange(0, xlim[1], 10))
	# ax.set_yticks(np.arange(0, ylim[1], 10))

	plt.tight_layout(pad=2.5)
	if not os.path.isdir(os.path.dirname(save_at)):
		os.makedirs(os.path.dirname(save_at))
	plt.savefig(save_at)
	if is_show:
		plt.show()

	print ("Save file at:", "{0}".format(save_at))
	release_mem(fig)


def joint_plot_2(x, y, xlabel, ylabel, xlim, ylim, title, save_at):

	df = pd.DataFrame()
	df[xlabel] = x
	df[ylabel] = y
	
	X = df[[xlabel,ylabel]].values
	gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
	y_pred = gmm.predict(X)
	
	# colors = [convert[k] for k in stable_lbl]
	fig, ax=plt.subplots(figsize=(8, 8))

	# Draw contour
	x_c = np.linspace(xlim[0], xlim[1], 100)
	y_c = np.linspace(ylim[0], ylim[1], 100) #d6-s2
	X, Y = np.meshgrid(x_c, y_c)
	XX = np.array([X.ravel(), Y.ravel()]).T
	Z = - gmm.score_samples(XX)
	Z = Z.reshape(X.shape)

	CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
									colors="grey", #linestyles="dashdot",
									levels=np.logspace(0, 3, 15))
	ax.clabel(CS, inline=1, fontsize=20)
	plt.pcolormesh(X, Y, Z, cmap = plt.get_cmap('Oranges'), # Greys, Oranges
									alpha=0.7
									) # Greens
	# hot, coolwarm, bwr, OrRd, Greys, GnBu, plasma, summer
	# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#     plt.yticks(np.arange(np.min(y), np.max(y) + 0.5, 1.0)) #d6-s2
	plt.scatter(x, y, color="grey",# label=state, 
                alpha=0.6, s=100, marker="o",
                linewidths=0.1, edgecolors=None)
	plt.xlabel(r'%s' %xlabel, **axis_font)
	plt.ylabel(r'%s' %ylabel, **axis_font)
	plt.yticks(np.arange(0, ylim[1], 10)) # # Ptdens: 10, valence: 0.5, Pt-Pt: 2
	plt.xticks(np.arange(0, xlim[1], 10))
	plt.title(title, **title_font)

	print(np.min(y), np.max(y), np.min(x), np.max(x))
	ax_setting(ax)
	plt.tight_layout(pad=1.1)
	makedirs(save_at)
	plt.savefig(save_at)
	print("Save at:", save_at)
	release_mem(fig)


def ax_histfill(x, label, ax, lbx, ubx, orientation="horizontal"):


	y_plot, x_plot, patches = ax.hist(x, bins=100, histtype='bar', # step, stepfilled, 'bar', 'barstacked'
				density=True, label=label, log=False,  
				color='grey', #edgecolor='none',
				alpha=0.5, linewidth=2, orientation=orientation)

	# abs0_reg = np.where(np.abs(x_plot) < abs_zero)[0]

	if lbx is not None and ubx is not None:	
		selected_reg = np.where((x_plot > lbx) & (x_plot < ubx))[0]

		for i in selected_reg:
			patches[i].set_color('orange')

	return ax

def joint_plot_fill(x, y, xlabel, ylabel, save_at, lbx, ubx, lby, uby):
	fig = plt.figure(figsize=(8, 8))

	grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
	main_ax = fig.add_subplot(grid[1:, :-1])
	y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
	x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
	
	sns.palplot("muted")
	sns.set_style('ticks')

	df = pd.DataFrame()
	df[xlabel] = x
	df[ylabel] = y
	# print (this_df)
	# g = sns.jointplot(df[xlabel], df[ylabel],
 #                     kind="kde", #ax=main_ax,
 #                     color="green",
 #                     # cmap="jet",
 #                     stat_func=None, n_levels=20, # GnBu
 #                     shade=True)
	g = sns.JointGrid(df[xlabel], df[ylabel], ratio=100,
				# kind="kde", shade=True, 
				# color="green",
				xlim=xlim, ylim=ylim)
	g.plot_joint(sns.kdeplot)
	g.ax_marg_x.set_axis_off()
	g.ax_marg_y.set_axis_off()

	# sns.jointplot(x, y,kind="kde", shade=True)

	plt.tight_layout(pad=1.1)

	makedirs(save_at)
	plt.savefig(save_at)
	release_mem(fig=fig)

def get_subaxis():
	fig = plt.figure(figsize=(8, 8))
	grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
	main_ax = fig.add_subplot(grid[1:, :-1])
	x_axis = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
	y_axis = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
	sns.set_style('ticks')


	return main_ax, x_axis. y_axis

def scatter_3d(points, save_at, label, color):
	xs, ys, zs = points[0], points[1], points[2] #[:, 0], points_T[:, 1], points_T[:, 2]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(xs, ys, zs, marker="o", color=color)
	# ax.title(label)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	makedirs(save_at)
	plt.savefig(save_at)
























