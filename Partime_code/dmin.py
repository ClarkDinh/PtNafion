import os
import shutil
import pandas as pd
import numpy as np
import math

from constant import *
from rolls import roll_full_k
# from plot_density import plot_density, makedirs
# from preprocess import get_img_idx_given_z, get_xyz

import multiprocessing
from functools import partial


def get_surf_lbl(main_cat, sub_cat):
	"""
	NOTICE
	This function is temporary moved to preprocess.py/get_surf_lbl
	The function in preprocess.py called by main.py/plot_mix_hists_scdata to
	create surface label dataset like "csv_surf_dist_morphology_fresh, 21k, 34k"
	Please do not call this func unless surface labels are estimated by Pt/Co density or Pt valence

	"""
	# store all z_raw files
	csv_dir = "{0}/csv/{1}/{2}".format(input_dir, main_cat, sub_cat)
	list_zfiles = os.listdir(csv_dir)
	list_zfiles.sort() 

	f1 = lambda x, y, min_value: x - y if x - y > min_value else min_value
	f2 = lambda x, y, max_value: x + y if x + y < max_value else max_value

 
	env_list = []
	# loc_idx_all = []
	# scan for all points, but layer by layer
	for z in range(z_size): 
		n_neighbors = 1
		z_minus = f1(z, n_neighbors, 0) # lower bound for z
		z_plus = f2(z, n_neighbors, z_size) # upper bound for z

		# get nearest neighbor by z dir
		z_env_all_df = [list_zfiles[t] for t in range(z_minus, z_plus)]
		if z_plus < z_size:
			z_env_all_df.append(list_zfiles[z_plus])

		
		this_env = None
		for z_file in z_env_all_df:
			z_df = pd.read_csv(str(input_dir + "/csv/" + main_cat + "/" + sub_cat + "/" + z_file), index_col=0)
			z_mt = z_df.values
			# z_rolls = roll(z_mt, k_step=n_neighbors) # shift for all x and y
			z_rolls = roll_full_k(z_mt, k_steps=n_neighbors) # shift for all x and y

			if this_env is None:
				this_env = z_rolls
			else:
				this_env = np.concatenate((this_env, z_rolls), axis=-1)
			# print ("z:", z, "z_rolls.shape:", z_rolls.shape, "len(z_rolls):", len(z_rolls))
		# print (this_env, this_env.shape)
		env_list.append(this_env) # test lai doan them nay, done, roll_full_k solved
		
		# means = np.nanmean(this_env, axis=-1)

		check_nan = np.isnan(np.sum(this_env, axis=-1))
		surf_lbl = list(map(lambda x: 1 if x else 0, check_nan))
		
		
		# z_df_out = pd.DataFrame(np.array(surf_lbl).reshape(y_size, x_size), index=z_df.index, columns=z_df.columns) 
		# plot_density(z_df_out.values, save_at="{0}/tmp.pdf".format(cwd),  cmap_name='jet', vmin=0, vmax=0.0003)
		

		this_zidx, this_zloc_idx = get_img_idx_given_z(zval=z) # # remove it
		z_df_out = pd.DataFrame(surf_lbl, index=this_zloc_idx, columns=["is_surface"]) #

		positions = np.array(list(map(lambda x: get_xyz(x), this_zloc_idx)))
		z_df_out["x"] = positions[:,0]
		z_df_out["y"] = positions[:,1]
		z_df_out["z"] = positions[:,2]

		save_at = "{0}/csv_surf_dist_{1}_test/layer_{2}.csv".format(result_dir, main_cat, z)
		makedirs(save_at)
		z_df_out.to_csv(save_at)
		print (z)
		# break

def poolcontext(*args, **kwargs):
	pool = multiprocessing.Pool(*args, **kwargs)
	yield pool
	pool.terminate()


def get_index_submatrix_3d(r, c, z, n_neighbor,
						maxrow, maxcol, max_z):


	f1 = lambda x, y, min_value: x - y if x - y > min_value else min_value
	north = f1(r, n_neighbor, 0)
	west = f1(c, n_neighbor, 0)
	up = f1(z, n_neighbor, 0)

	f2 = lambda x, y, max_value: x + y if x + y < max_value else max_value
	south = f2(r, n_neighbor, maxrow)
	east = f2(c, n_neighbor, maxcol)
	down = f2(z, n_neighbor, max_z)

	return east, west, north, south, up, down






def chec_surf_in_enf(env_df, center_vec):
	center_vec = np.array(center_vec)
	if len(env_df) != 0: # if env_df is non-empty set
		env_idxes = env_df.index
		# find min distance and nearest surf point
		for this_i, env_idx in enumerate(env_idxes):
			this_x = env_df.loc[env_idx, "x"]
			this_y = env_df.loc[env_idx, "y"]
			this_z = env_df.loc[env_idx, "z"]
			this_vec = np.array([this_x, this_y, this_z])
			this_d = np.linalg.norm(this_vec - center_vec)
			if this_i == 0:
				dmin = this_d

			if this_d < dmin:
				dmin = this_d
				nearest_point = this_vec
	else:
		dmin = np.nan
	return dmin



def get_d2surf(x, y, z, list_zfiles, n_neighbor):

	east, west, north, south, z_up, z_down = get_index_submatrix_3d(r=int(y), c=int(x), z=int(z), 
				n_neighbor=n_neighbor,
				maxrow=y_size, maxcol=x_size,
				max_z=z_size)

	z_env_all_df = [list_zfiles[t] for t in range(z_up, z_down)]
	if z_down < z_size:
		z_env_all_df.append(list_zfiles[z_down])

	# loop for all z layers
	dmins = []
	for z_file in z_env_all_df:
		z_df = pd.read_csv("{0}/csv_surf_dist/{1}".format(result_dir, z_file), index_col=0)

		surf_df = z_df[z_df["is_surface"] == 1]

		env_df = surf_df[(((surf_df["x"] >= west) & (surf_df["x"] <= east))
							 & ((surf_df["y"] >= north) & (surf_df["y"] <= south))
							 & ((surf_df["z"] >= z_up) & (surf_df["z"] <= z_down)))]

		dmin = chec_surf_in_enf(env_df=env_df, center_vec=[x, y, z])
		dmins.append(dmin)

	return dmins


def get_dmin(bidx, bulk_df, list_zfiles):
	x, y, z = bulk_df.loc[bidx, ["x", "y", "z"]]
	found = False
	n_neighbor = 1
	while not found:
		dmins = get_d2surf(x=x, y=y, z=z, list_zfiles=list_zfiles, n_neighbor=n_neighbor)
		dmin_val = np.nanmin(dmins)
		print (x, y, z, "n_neighbor:",n_neighbor, "dmins:", dmins, "dmin_val:", dmin_val)
		if math.isnan(dmin_val):
			n_neighbor += 1
		else:
			# df.loc[bidx, "dist2surf"] = dmin_val
			found = True
	return dmin_val

def get_dmins():
	csv_dir = "{0}/csv_surf_dist".format(result_dir)
	list_zfiles = os.listdir(csv_dir)
	list_zfiles.sort() 


	for z in range(z_size): 
		file_name = "{0}/csv_surf_dist/layer_{1}.csv".format(result_dir, z)
		df = pd.read_csv(file_name, index_col=0)

		df["dist2surf"] = 0
		
		bulk_df = df[df["is_surface"] == 0]
		bulk_idxs = list(bulk_df.index)

		print (bulk_idxs[:10])
		for bidx in bulk_idxs:
			x, y, z = bulk_df.loc[bidx, ["x", "y", "z"]]
			found = False
			n_neighbor = 1
			while not found:
				dmins = get_d2surf(x=x, y=y, z=z, list_zfiles=list_zfiles, n_neighbor=n_neighbor)
				dmin_val = np.nanmin(dmins)
				print (x, y, z, "n_neighbor:",n_neighbor, "dmins:", dmins, "dmin_val:", dmin_val)
				if math.isnan(dmin_val):
					n_neighbor += 1
				else:
					df.loc[bidx, "dist2surf"] = dmin_val
					found = True
		# east, west, north, south, up, down
		
		# pool = multiprocessing.Pool(processes=128)
		# results = pool.map(partial(get_dmin, bulk_df=bulk_df, list_zfiles=list_zfiles), bulk_idxs)
		# # results = 0
		# df.loc[bulk_idxs, "dist2surf"] = results


		# run for layer_0 first
		save_at = "{0}/csv_dist2surf/layer_{1}.csv".format(result_dir, z)
		makedirs(save_at)
		df.to_csv(save_at)
		break


if __name__ == "__main__":
	sub_cats = dict({
		"Co_density": ["e_Co_fresh", "e_Co_21k", "e_Co_34k"],
		"Pt_density": ["e_Pt_fresh", "e_Pt_21k", "e_Pt_34k"],
		"morphology": ["fresh", "21k", "34k"],
		"Pt_valence": ["v_04_fresh", "v_04_21k", "v_04_34k", "v_10_fresh", "v_10_21k", "v_10_34k"],
		"delta density": ["d_Co", "d_Pt"]

		})
	get_surf_lbl(main_cat="morphology", sub_cat="21k") 
	# Pt_density, e_Pt_fresh
	# Co_density, e_Co_fresh

	get_dmins()


	# get_xyz(index="x291_y10_z3")
