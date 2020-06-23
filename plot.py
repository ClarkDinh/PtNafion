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
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
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



def ax_setting():
	plt.style.use('default')
	plt.tick_params(axis='x', which='major', labelsize=13)
	plt.tick_params(axis='y', which='major', labelsize=13)


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


def plot_density(values, save_at,  cmap_name="Oranges", vmin=None, vmax=None, is_save2input=False):
	# input: matrix [n_rows, n_cols] of any value
	# output: figure
	fig = plt.figure(figsize=(10, 10))

	if vmin is None and vmax is None:
		max_abs = 0.95 * max((abs(np.min(values)), abs(np.max(values))))
		vmin = -max_abs
		vmax = max_abs




	# m = np.ma.masked_where(np.isnan(values),values)
	# plt.pcolor(m)
	# img = plt.imread(values, format='raw')

	cmap = plt.get_cmap(cmap_name)
	cmap.set_bad('white')
	# img = cmap(values)
	# print (img)
	plt.imshow(values, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)

	plt.colorbar()
	plt.xlabel('x', **axis_font)
	plt.ylabel('y', **axis_font)

	title = save_at[save_at.find('result/') + len('result/'): ]
	plt.title(title, **title_font)


	makedirs(save_at)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	plt.savefig(save_at, transparent=False)
	print ("Save file at:", save_at)

	# # save to input
	if is_save2input:
		save_txt = save_at.replace("result", "input").replace(".pdf", ".txt")
		makedirs(save_txt)
		np.savetxt(save_txt, values)

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

	# save all predicted val to csv 

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






def joint_plot(x, y, xlabel, ylabel, xlim, ylim, title, save_at):
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
	cfset = ax.contourf(xx, yy, f, cmap='jet')
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


	# x_hist = g.ax_marg_x
	# y_hist = g.ax_marg_y
	# x_hist = ax_histfill(x=x, label=xlabel, ax=x_hist, lbx=lbx, ubx=ubx, orientation='vertical')
	# y_hist = ax_histfill(x=y, label=ylabel, ax=y_hist, lbx=lby, ubx=uby, orientation='horizontal')
	# main_ax.tick_params(axis='both', labelsize=10)
	# main_ax.set_xlabel(xlabel, **axis_font)
	# main_ax.set_ylabel(ylabel, **axis_font)

	# plt.setp(x_hist.get_xticklabels(), visible=False)
	# plt.setp(y_hist.get_yticklabels(), visible=False)
	

	# sns.jointplot(x, y,kind="kde", shade=True)

	plt.tight_layout(pad=1.1)

	makedirs(save_at)
	plt.savefig(save_at)
	release_mem(fig=fig)



class corr_analysis():
	def __init__(self, data_file, threshold_pearson, threshold_spearman, all_variable, out_dir):
		self.df = pd.read_csv(filepath_or_buffer=data_file, index_col=0)
		#self.variables = list(self.df.columns)
		if all_variable:
			self.variables = self.df.columns
		else:
			self.variables = all_variable


		self.instance_name = self.df.index
		self.threshold_pearson = threshold_pearson
		self.threshold_spearman = threshold_spearman
		self.out_dir = out_dir

		self.axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}

		self.title_font = {'fontname': 'serif', 'size': 14}

	def plot_correlation(self, x, y, xlabel, ylabel, title):
		fig = plt.figure(figsize=(16, 16))
		ax_setting()
		# plt.scatter(x=x, y=y, s=30, c='blue', alpha=0.1, edgecolors='none')
		sns.jointplot(x, y,  kind="kde", color="g")
		plt.xlabel(r'%s' %xlabel, **self.axis_font)
		plt.ylabel(r'%s' %ylabel, **self.axis_font)
		plt.title(title, **self.axis_font)
		plt.savefig("%s/%s vs %s.pdf" % (self.out_dir, xlabel, ylabel))
		release_mem(fig)

	def correlation_matrix(self):

		all_var_pair = itertools.combinations(self.variables, r=2)

		this_df_pearson = pd.DataFrame(index=self.variables, columns=self.variables)
		this_df_spear = pd.DataFrame(index=self.variables, columns=self.variables)

		for pair in all_var_pair:

			x = self.df[pair[0]]
			y = self.df[pair[1]]
			pearson, p_value = stats.pearsonr(x=x, y=y)
			spear1, p_value = stats.spearmanr(a=x, b=y)
			spear2, p_value = stats.spearmanr(a=y, b=x)

			# if pearson > threshold_pearson:
			self.plot_correlation(x=x, y=y, 
				xlabel=pair[0], ylabel=pair[1], 
				title="Pearson {0}".format(round(pearson, 3)))

			pearson = abs(pearson)
			spear1 = abs(spear1)
			spear2 = abs(spear2)

			if pearson > self.threshold_pearson:
				this_df_pearson[pair[0]][pair[1]] = pearson
				this_df_pearson[pair[1]][pair[0]] = pearson
			else:
				this_df_pearson[pair[0]][pair[1]] = 0
				this_df_pearson[pair[1]][pair[0]] = 0

			if spear1 > self.threshold_spearman:
				this_df_spear[pair[0]][pair[1]] = spear1
			else:
				this_df_spear[pair[0]][pair[1]] = 0

			if spear2 > self.threshold_spearman:
				this_df_spear[pair[1]][pair[0]] = spear2
			else:
				this_df_spear[pair[1]][pair[0]] = 0

			this_df_pearson[pair[0]][pair[0]] = 1
			this_df_pearson[pair[1]][pair[1]] = 1
			this_df_spear[pair[0]][pair[0]] = 1
			this_df_spear[pair[1]][pair[1]] = 1


		this_df_pearson.to_csv("%s/Pearson.csv" %self.out_dir)
		this_df_spear.to_csv("%s/Spearman.csv" %self.out_dir)

	def plot_task3(self):

		self.correlation_matrix()

		for type_ in ["Pearson", "Spearman"]:
			X = pd.read_csv("%s/%s.csv" %(self.out_dir, type_), index_col=0)
			X_MDS = X.copy()

			X = X[X > 0.7]

			fig = plt.figure(figsize=(16, 16))
			ax = sns.heatmap(X)


			for item in ax.get_yticklabels():
				item.set_fontsize(8)
				item.set_rotation(0)
				item.set_fontname('serif')

			for item in ax.get_xticklabels():
				item.set_fontsize(8)
				item.set_fontname('serif')
				item.set_rotation(90)

			plt.title("%s similarity" %(type_), **self.title_font)
			plt.savefig("%s/%s.eps" %(self.out_dir, type_))
			release_mem(fig)







def get_subaxis():
	fig = plt.figure(figsize=(8, 8))
	grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
	main_ax = fig.add_subplot(grid[1:, :-1])
	x_axis = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
	y_axis = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
	sns.set_style('ticks')


	return main_ax, x_axis. y_axis





# def merge_2image(morph_df, df2, vmin, vmax, title, save_at, cmapdf2='jet'):
# 	# from matplotlib.colors import LightSource
# 	fig = plt.figure(figsize=(10, 10))


# 	plt.imshow(morph_df.values, cmap='binary')
# 	plt.colorbar()

# 	plt.imshow(df2.values, cmap=cmapdf2, interpolation='none', 
# 		vmin=vmin, vmax=vmax)

# 	# cmap.set_bad('white')
	
# 	# plt.colorbar()
# 	plt.xlabel('x', **axis_font)
# 	plt.ylabel('y', **axis_font)

# 	plt.title(title, **title_font)

# 	makedirs(save_at)
# 	plt.savefig(save_at)
