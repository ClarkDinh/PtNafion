
from Rubber_constant import *
from hierarchical_clustering import hac, show_tree
from plot import scatter_3d, plot_density
import matplotlib.pyplot as plt
import matplotlib.colors
import copy, sys
from random import random
import hdbscan
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import minmax_scale
import time



def extend_box(
	xmin, xmax, ymin, ymax, zmin, zmax, delta, init_layer, final_layer):
	xmin_ = max([xmin-delta, 0])
	ymin_ = max([ymin-delta, 0])
	zmin_ = max([zmin-delta, 0])

	xmax_ = min([xmax+delta, SIZE_X])
	ymax_ = min([ymax+delta, SIZE_Y])
	zmax_ = min([zmax+delta, final_layer-init_layer])
	return xmin_, xmax_, ymin_, ymax_, zmin_, zmax_



def parse2submatrix(org_data, xs, ys, zs, savedir, 
	pid, task, init_layer, final_layer, is_verbose):

	xmin, xmax = np.min(xs), np.max(xs)
	ymin, ymax = np.min(ys), np.max(ys)
	zmin, zmax = np.min(zs), np.max(zs)

	# box = np.full([50, 50, 50], np.nan) # [int(xmax-xmin), int(ymax - ymin), int(zmax - zmin)]
	size_x = xmax - xmin
	size_y = ymax - ymin
	size_z = zmax - zmin

	# print (org_data.shape)
	xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = extend_box(
		xmin, xmax, ymin, ymax, zmin, zmax, 
		delta=2, init_layer=init_layer, final_layer=final_layer)

	box = org_data[zmin_:zmax_, ymin_:ymax_, xmin_:xmax_]# [xmin:xmax, ymin:ymax, zmin:zmax]
	if is_verbose:
		print ("box.shape", box.shape)
		print (xmin_, xmax_, ymin_, ymax_, zmin_, zmax_)

	z_layers = box.shape[0]
	if z_layers != 0:
		vmin = np.min(box)
		vmax = np.max(box)

		shape = "{0}-{1}-{2}".format(box.shape[0], box.shape[1], box.shape[2])
		for ith in range(z_layers):
			data = box[ith]
			row_id = "/particle_{0}_{1}/".format(pid, shape)+"/r"+"{0:04}".format(init_layer+zmin_+ith)+"_xrange_{0}-{1}-_yrange_{2}-{3}".format(xmin, xmax, ymin, ymax)
			file = savedir+ row_id +".txt"
			save_layer2text(data, file=file)

			if is_verbose:
				figfile = file.replace(text_dir, fig_dir).replace(".txt", ".pdf")
				plot_density(values=data, save_at=figfile,  cmap_name="jet", 
						title=row_id, vmin=vmin, vmax=vmax, is_save2input=None)

	# scatter_3d(points=box, save_at=savedir+"/fig/particle_{}.pdf".format(pid), 
	# 		label=pid, color="jet")

def one_sample(x, y, z, final_layer, init_layer):
	x_sample = x + np.random.uniform(-0.5, 0.5)
	y_sample = y + np.random.uniform(-0.5, 0.5)
	z_sample = z + np.random.uniform(-0.5, 0.5)

	x_sample = min([x_sample, SIZE_X])
	y_sample = min([y_sample, SIZE_Y])
	z_sample = min([z_sample, final_layer-init_layer])

	x_sample = max([x_sample, 0])
	y_sample = max([y_sample, 0])
	z_sample = max([z_sample, 0])

	return [x_sample, y_sample, z_sample]

def convert_bulk_bkg(obj3d, org_obj3d, result_dir, job,
			init_layer, final_layer, task, kwargs, is_verbose):
	X = np.array(np.where(obj3d == 1.0)).T
	obj3d_nan = np.full(obj3d.shape, np.nan) # obj3d.shape
	

	# # for HAC only
	# kwargs = dict({"distance_threshold":None,
	# 		"n_clusters":200, # either "distance_threshold" or "n_clusters" will be set
	# 		"affinity":'euclidean', "linkage":'ward', 
	# 		})
	# hac_model = hac(**kwargs)
	# hac_model.fit(X)
	# saveat = result_dir+"/dedogram.pdf"
	# labels = hac_model.model.labels_
	# # # for HAC ploting only
	# dd_plot_kwargs = dict({"truncate_mode":'level', "p":3})
	# dendrogram = hac_model.plot_dendrogram(saveat=saveat, **dd_plot_kwargs)
	# # print (dendrogram["ivl"])

	layer_id = "/layer_{0}-{1}".format(init_layer, final_layer)

	by_particle_dir = result_dir+text_dir+"/particles/"+job+layer_id
	print ("Prepare to fit")
	clusterer = hdbscan.HDBSCAN(min_cluster_size=kwargs["min_cluster_size"], 
		min_samples=kwargs["min_samples"], 
		alpha=kwargs["alpha"], allow_single_cluster=kwargs["allow_single_cluster"],
		prediction_data=True)
	# X_fit = copy.copy(X)
	# density = []
	# points = np.array(X)
	# for p in points:
	# 	pz, py, px = p[0], p[1], p[2]
	# 	print ("pz, py, px", pz, py, px)
	# 	n_upsampling = int(org_obj3d[pz][py][px]*1000)
		# if n_upsampling > 0:
		# 	# print ("Upsampling: ", n_upsampling, "points")
		# 	for i in range(n_upsampling):
		# 		p_sample = one_sample(px, py, pz, final_layer, init_layer)
		# 		X_fit = np.append(X_fit, [p_sample], axis=0)

	# print ("density:", density)
	# X_fit = np.append(X_fit, np.array(density), axis=1)
	
	# X_fit = minmax_scale(X_fit)

	clusterer.fit(X)
	print ("Done fitting")

	# labels = clusterer.predict(X)

	labels, strengths = hdbscan.approximate_predict(clusterer, X)

	assert X.shape[0] == len(labels)

	if is_verbose:
		# # # plot dendrogram
		fig = plt.figure(figsize=(10, 10), dpi=300)
		clusterer.condensed_tree_.plot(select_clusters=True,
	        selection_palette=sns.color_palette('deep', 8))
		save_at = by_particle_dir+"dendrogram.pdf"
		makedirs(save_at)
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		plt.savefig(save_at, transparent=False)
		print ("Save file at:", save_at)

		# # # End plot dendrogram

	set_labels = set(labels)
	n_clusters = len(set_labels)

	cmap = plt.cm.rainbow
	norm = matplotlib.colors.Normalize(vmin=0.0, vmax=n_clusters)

	for lbl in set_labels:
		pos_of_lbl = X[np.where(labels==lbl)]
		# print (pos_of_lbl)

		zs, ys, xs  = pos_of_lbl[:, 0], pos_of_lbl[:, 1], pos_of_lbl[:, 2]
		obj3d_nan[zs, ys, xs] = lbl #[lbl] * len(pos_of_lbl)

		print ("lbl:", lbl, "total:", len(set_labels), "particle size:", len(xs), len(ys), len(zs))
		parse2submatrix(org_data=org_obj3d, 
			xs=xs, ys=ys, zs=zs, savedir=by_particle_dir,
			pid=lbl, task=task,
			init_layer=init_layer, final_layer=final_layer,
			is_verbose=is_verbose)
		# break		 

	label_dir = result_dir+text_dir+"/lbl_in3D/"+job+layer_id
	label_fig_dir = result_dir+fig_dir+"/lbl_in3D/"+job+layer_id

	n_layers = obj3d.shape[0]

	vmin = np.nanmin(obj3d_nan)
	vmax = np.nanmax(obj3d_nan)

	for ith, layer in enumerate(range(init_layer, final_layer+1)):
		data = obj3d_nan[ith]
		file = label_dir+"/lbl_{}.txt".format(layer)
		makedirs(file)
		save_layer2text(data, file=file)
		print ("Save at:", file)

		if is_verbose:
			figfile = label_fig_dir+"/lbl_{}.pdf".format(layer)
			plot_density(values=data, save_at=figfile,  cmap_name="jet", 
					title=layer, vmin=vmin, vmax=vmax, is_save2input=None,
					is_lbl=True,set_labels=set_labels)



def main(kwargs, job):
	

	tmp = "tmp.pkl"
	init_layer = 0 # 120
	step = 546 # 546
	task = "bkg"
	
	prefix = get_prefix(kwargs)

	for init_layer in range(init_layer, SIZE_Z[task], step):
		final_layer = min([init_layer + step, SIZE_Z[task]]) 
		
		obj3d, all_min, all_max = cat2Dto3D(jobdir="label_txt/"+job, 
			init_layer=init_layer, final_layer=final_layer)

		org_obj3d, all_min, all_max = cat2Dto3D(jobdir="txt/"+job, 
			init_layer=init_layer, final_layer=final_layer)

		print ("min(all_min): ", np.min(all_min))
		print ("max(all_max): ", np.max(all_max))

		convert_bulk_bkg(obj3d=obj3d, 
			result_dir=ResultDir+prefix, job=job,
			org_obj3d=org_obj3d, task=task,
			init_layer=init_layer, final_layer=final_layer,
			kwargs=kwargs, is_verbose=False)
		break

def param_check(kwargs_list, job):
	df = pd.DataFrame(columns=list(kwargs_list[0].keys())+["n_particles"])
	for kwargs in kwargs_list:
		prefix = get_prefix(kwargs)
		
		result_dir = ResultDir+prefix
		by_particle_dir = result_dir+text_dir+"/particles/"+job+"/layer_120-140"

		particles = get_subdirs(sdir=by_particle_dir)

		for k, v in kwargs.items():
			df.loc[prefix, k] = v
		df.loc[prefix, "n_particles"] = len(particles)

		print (prefix, len(particles))

		# break
	df.to_csv(ResultDir+"/params.csv")




if __name__ == "__main__":

	job = "Whole/fresh_CT13/bkg"


	# # 1. test with one kwargs
	t1 = time.time()
	pr_file = sys.argv[-1]
	kwargs = load_pickle(filename=pr_file)
	print (kwargs)
	
	# # 3. prepare for batch run
	main(kwargs=kwargs, job=job)
	t2 = time.time()
	print ("========")
	print (kwargs)
	kwargs["duration"] = t2 - t1
	saveat = pr_file.replace("/params_grid/", "/time_estimation/")
	# makedirs(saveat)
	dump_pickle(kwargs, saveat)
	print ("Duration:", t2 - t1)
	print ("========")
	print ("Done.")

	# # 4. param check
	# param_check(kwargs_list=kwargs_list, job=job)





