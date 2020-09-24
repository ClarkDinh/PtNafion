import numpy as np
import matplotlib.pyplot as plt
import time, gc, os, itertools 
import pandas as pd

from plot import plot_density, plot_hist, makedirs, joint_plot_1
from Nafion_constant import *
from pylab import * 
from fine_tunning import plot_joinmap
import copy
# from fine_tunning import get_dmin
path = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion/input/2020-06-17-NewRequest/txt/"

def redox_lbl(fixT, fixP, fixV, diff_state, task="diff_p"):
	final_state, init_state = diff_state 

	if task == "diff_p":
		fix_val = "{0}{1}".format(fixT, fixV)
	if task == "diff_v":
		fix_val = "{0}{1}".format(fixT, fixP)
	if task == "diff_t":
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


	redox_sum = sum(redox_states, axis=0)
	# # filter crack area
	# if task == "diff_t":
	print (init_state)
	bulk_label_file = path+"feature/task3/num_bulk_label"+fixP+fixT+fixV+"_morphology.txt"
	bulk_label_lbl = np.loadtxt(bulk_label_file)
	# crack_id = np.where(bulk_label_lbl==-1) 
	redox_sum = np.array(redox_sum)
	redox_sum[bulk_label_lbl==-1] = np.nan # np.nan
	print ("Shape before: ", redox_sum.shape)
	# # delete 10 columns on the left

	redox_sum_copy = copy.copy(redox_sum)
	redox_sum_copy = np.delete(redox_sum_copy, index2remove, 1)  # delete second column of C
	print ("Shape after: ", redox_sum_copy.shape)

	# # to plot
	vmin = 1
	vmax = 8 
	cmap_name="jet"
	# cmap_name = cm.get_cmap('PiYG', 8)


	prefix = "{0}/redox/{1}/{2}_{3}___{4}".format(result_dir,
		task, fix_val, final_state,  init_state) 

	save_at = prefix + ".pdf"

	redox_file_save_at = "{0}/redox/{1}/{2}_{3}___{4}.txt".format(input_dir,
		task, fix_val, final_state,  init_state)
	plot_density(values=redox_sum_copy, save_at=save_at,  # diff_PtDens_lbl
		title=save_at.replace(result_dir,""),
		cmap_name=cmap_name, vmin=vmin, vmax=vmax, 
		is_save2input=redox_file_save_at)

	# # save to txt
	save_txt = prefix.replace("result", "input") + ".txt"
	makedirs(save_txt)

	np.savetxt(save_txt, redox_sum_copy) # redox_sum_copy, redox_sum

	for i, rst in enumerate(redox_states):
		tmp_save_at = "{0}/redox_{1}.pdf".format(prefix, i+1)
		plot_density(values=rst, save_at=tmp_save_at,
		title=tmp_save_at.replace(result_dir,""),
			cmap_name=cmap_name, vmin=vmin, vmax=vmax)


def join_redox_dmin(fixT, fixP, fixV, diff_state, dmin_measure, task="diff_p"):
	final_state, init_state = diff_state # # here is diff voltage


	if task == "diff_p":
		fix_val = "{0}{1}".format(fixT, fixV)
	if task == "diff_v":
		fix_val = "{0}{1}".format(fixT, fixP)
	if task == "diff_t":
		fix_val = "{0}{1}".format(fixP, fixV)

	# morph_file = input_dir+"/feature/task3/bulk_label_"+fixP+fixT+fixV+"_morphology.txt"
	# morph_val = np.loadtxt(morph_file, dtype=str).ravel()
	# bkg_idx = np.where(morph_val=="bkg")[0]

	# # get dmin
	dmin_value = get_dmin(fixP, fixT, fixV)
	dmin_value_copy = copy.copy(dmin_value)
	bkg_idx = np.where(dmin_value==-50)[0]

	# dmin_value = np.loadtxt(dmin_file)

	# # get redox value
	redox_file = "{0}/redox/{1}/{2}_{3}___{4}.txt".format(input_dir,
		task, fix_val, final_state, init_state)
	redox_label = np.loadtxt(redox_file)
	# redox_label = np.delete(redox_label, index2remove, 1)  
	# redox_label = redox_label.ravel()
	print ("compare shape", dmin_value.shape, redox_label.shape)
	# joint_plot(x=dmin_value.ravel(), y=redox_label.ravel(), 
	# 	xlabel="dmin_{0}{1}{2}".format(fixP, fixT, fixV), ylabel="redox_state", 
	# 	xlim=[-2, 40], ylim=None,
	# 	title=save_at.replace(result_dir, ""),
	# 	save_at=save_at)

	xlabel = "dmin"
	ylabel = "redox"

	# xlabel = "redox"
	# ylabel = "dmin"
	xlim=[-2, 50]
	ylim=None

	save_at = "{0}/dmin_redox/{1}/dmin_at{5}/{2}_{3}___{4}_redox.pdf".format(result_dir,
		task, fix_val, final_state, init_state, dmin_measure) # final_state or init_state
	df_redox_dmin = remove_nan(ignore_first=bkg_idx,
			matrix1=dmin_value_copy,matrix2=redox_label,lbl1=xlabel,lbl2=ylabel)

	plot_joinmap(df_redox_dmin, selected_inst=None, xlbl=xlabel, ylbl=ylabel, 
					xlbl_2fig="Distance to surface", ylbl_2fig="Redox state", color="blue",
					save_at=save_at, 
					is_gmm_xhist=False, is_gmm_yhist=False, 
					means=None, weight=None, cov_matrix=None, 
					n_components=None, xlim=xlim, ylim=ylim,
					main_ax_type="candle")


def save_diff_2csv(df, fixT, fixP, fixV, diff_state, save_at, task="diff_p"):
	final_state, init_state = diff_state 

	if task == "diff_p":
		fix_val = "{0}{1}".format(fixT, fixV)
	if task == "diff_v":
		fix_val = "{0}{1}".format(fixT, fixP)
	if task == "diff_t":
		fix_val = "{0}{1}".format(fixP, fixV)

	# # save diff value
	failed_feature = []
	for consider_Ft in Features:
		prefix_input = fix_val + consider_Ft

		diff_val_file = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					final_state, init_state)
		
		diff_val, is_diff_val_pos = pos_neg_lbl_cvt(inputfile=diff_val_file)

		feature_name = diff_val_file.replace(input_dir, "")
		try:
			df[feature_name] = diff_val.ravel()
		except Exception as e:
			failed_feature.append(diff_val_file)
			pass


	# # save redox value
	redox_file = "{0}/redox/{1}/{2}_{3}___{4}.txt".format(myinput_dir,
		task, fix_val, final_state, init_state) 
	redox_value = np.loadtxt(redox_file)
	feature_name = redox_file.replace(myinput_dir, "")
	df[feature_name] = redox_value.ravel()

	# # save origin feature
	
	origin_fts = list(itertools.product(*[Features, Positions, Voltages, Times]))
	for origin_ft in origin_fts:
		ft, p, v, t = origin_ft
		this_dir = "{0}/{1}/{2}/{3}{4}_{2}.txt".format(myinput_dir, p, ft, t, v)
		print (this_dir)

		this_value = np.loadtxt(this_dir)

		this_feature = this_dir.replace(myinput_dir, "")
		df[this_feature] = this_value.ravel()
	makedirs(save_at)
	df.to_csv(save_at)
	return df, failed_feature


if __name__ == "__main__":
	result_dir = "{}/result/0916_response".format(maindir)

	# # in considering diff_t
	tasks = ["diff_v"] # "diff_p", "diff_t", "diff_v"
	is_redox_lbl = True
	is_redox2dmin = True
	is_save2csv = False

	if False:
		fixT = None
		fixP = None
		fixV = None
		task = "diff_t"
		a = [dT, Positions, Voltages]
		combs = list(itertools.product(*a))
		for comb in combs:
			diff_T, fixP, fixV = comb
			# print (diff_state, fixP, fixV)
			print (comb)
			# redox_diff_t(fixP=fixP, fixV=fixV, diff_state=diff_state, task=task)
			redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_T, task=task)

			# break
	
	if False:
		fixT = None
		fixP = None
		fixV = None
		task = "diff_v"
		a = [Times, Positions, dV]
		combs = list(itertools.product(*a))
		for comb in combs:
			fixT, fixP, diff_V = comb
			print (comb)
			# redox_diff_v(fixT=fixT, fixP=fixP, diff_state=diff_V, task=task)
			redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, 
				diff_state=diff_V, task=task)
			# break

	if is_redox_lbl:
		for task in tasks:
			fixT = None
			fixP = None
			fixV = None
			if task == "diff_p":
				a = [Times, dP, Voltages]
			if task == "diff_v":
				a = [Times, Positions, dV]
			if task == "diff_t":
				a = [dT, Positions, Voltages]

			combs = list(itertools.product(*a))
			for comb in combs:
				if task == "diff_p":
					fixT, diff_P, fixV = comb
					fixP = diff_P[1]
					redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, 
						diff_state=diff_P, task=task)

				if task == "diff_v":
					fixT, fixP, diff_V = comb
					fixV = diff_V[1]
					redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, 
						diff_state=diff_V, task=task)
		
				if task == "diff_t":
					diff_T, fixP, fixV = comb
					fixT = diff_T[1]
					redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, 
						diff_state=diff_T, task=task)


	if is_redox2dmin:
		for task in tasks:

			if task == "diff_p":
				a = [Times, dP, Voltages]
			if task == "diff_v":
				a = [Times, Positions, dV]
			if task == "diff_t":
				a = [dT, Positions, Voltages]

			combs = list(itertools.product(*a))
			for comb in combs:
				if task == "diff_p":
					fixT, diff_P, fixV = comb
					fixP = diff_P[1]
					join_redox_dmin(fixT=fixT, fixP=fixP, fixV=fixV, dmin_measure=fixP, diff_state=diff_P, task=task)

				
				if task == "diff_v":
					fixT, fixP, diff_V = comb
					fixV = diff_V[1]
					join_redox_dmin(fixT=fixT, fixP=fixP, fixV=fixV, dmin_measure=fixV, diff_state=diff_V, task=task)
		
				if task == "diff_t":
					diff_T, fixP, fixV = comb
					fixT = diff_T[1]
					join_redox_dmin(fixT=fixT, fixP=fixP, fixV=fixV, dmin_measure=fixT, diff_state=diff_T, task=task)

			# 	break
			# break


	if is_save2csv:
		df = pd.DataFrame()
		_x = np.arange(0, size_x, 1)
		_y = np.arange(0, size_x, 1)
		merged = np.array(list(itertools.product(_x, repeat=2))).T

		df["x"] = merged[0]
		df["y"] = merged[1]
		save_at = "{0}/PtNafion.csv".format(myinput_dir)
		# if False:

		all_failed_feature =[]
		for task in tasks:

			if task == "diff_p":
				a = [Times, dP, Voltages]
			if task == "diff_v":
				a = [Times, Positions, dV]
			if task == "diff_t":
				a = [dT, Positions, Voltages]

			combs = list(itertools.product(*a))
			for comb in combs:
				if task == "diff_p":
					fixT, diff_P, fixV = comb
					fixP = diff_P[0]
					df, failed_feature = save_diff_2csv(df=df, fixT=fixT, fixP=fixP, fixV=fixV, 
						diff_state=diff_P, save_at=save_at, task=task)
				if task == "diff_v":
					fixT, fixP, diff_V = comb
					fixV = diff_V[0]
					df, failed_feature = save_diff_2csv(df=df, fixT=fixT, fixP=fixP, fixV=fixV, 
						diff_state=diff_V, save_at=save_at, task=task)
		
				if task == "diff_t":
					diff_T, fixP, fixV = comb
					fixT = diff_T[0]
					df, failed_feature = save_diff_2csv(df=df, fixT=fixT, fixP=fixP, fixV=fixV, 
						diff_state=diff_T, save_at=save_at, task=task)
				print ("failed_feature", failed_feature)
				all_failed_feature.append(failed_feature)

			# 	break
			# break
	# print (all_failed_feature)

























