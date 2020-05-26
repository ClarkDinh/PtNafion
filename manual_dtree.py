import numpy as np
import matplotlib.pyplot as plt
import time, gc, os, itertools
import pandas as pd

from plot import plot_density, plot_hist, makedirs, joint_plot
from Nafion_constant import *
from pylab import *
from redox_label import pos_neg_lbl_cvt

total_px = 262144

def is_smaller_lbl_cvt(inputfile, top_k):
	try:
		data = np.loadtxt(inputfile)
		
	except Exception as e:
		revise_file = inputfile.replace("___", "____")
		data = np.loadtxt(revise_file)
		if not os.path.isfile(revise_file):
			print ("TOTALLY WRONG DIR HERE!", inputfile)
		pass

	# # sort ascending
	sorted_data = np.sort(data.ravel())
	lower_bound = sorted_data[int(top_k*total_px)]

	# # get label under conditions
	
	lbl = np.where(data<=lower_bound, True, False)

	# save_at = "{0}/tmp.pdf".format(result_dir) # for test
	# plot_density(values=diff_PtDens_val, save_at=save_at,  # diff_PtDens_lbl
	# 	cmap_name="bwr", vmin=None, vmax=None)

	return data, lbl

def mechanism(fixT, fixP, fixV, diff_state, task):
	final_state, init_state = diff_state 

	if task == "diff_p":
		fix_val = "{0}{1}".format(fixT, fixV)
		fixP = init_state # temporal assign for condition 2
	
	if task == "diff_v":
		fix_val = "{0}{1}".format(fixT, fixP)
		fixV = init_state # temporal assign for condition 2

	if task == "diff_t":
		fix_val = "{0}{1}".format(fixP, fixV)
		fixT = init_state # temporal assign for condition 2

	# # condition 1
	# # read diff_PtDens label
	consider_Ft = "Pt-density"
	prefix_input = fix_val + consider_Ft
	diff_PtDens = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					final_state, init_state)
	diff_PtDens_val, is_diff_PtDens_pos = pos_neg_lbl_cvt(inputfile=diff_PtDens, is_get_zero=True)
	cond1 = is_diff_PtDens_pos

	# # condition 2
	# # Pt Dens and PtPt
	top_k=0.1
	PtDens = get_org_dir(p=fixP, feature="Pt-density", t=fixT, v=fixV, ftype="txt")
	# PtDens_val, is_PtDens_pos = pos_neg_lbl_cvt(inputfile=PtDens)
	PtDens_val, is_PtDens_pos = is_smaller_lbl_cvt(inputfile=PtDens, top_k=top_k)

	PtPt = get_org_dir(p=fixP, feature="Pt-Pt", t=fixT, v=fixV, ftype="txt")
	# PtPt_val, is_PtPt_pos = pos_neg_lbl_cvt(inputfile=PtPt)
	PtPt_val, is_PtPt_pos = is_smaller_lbl_cvt(inputfile=PtDens, top_k=top_k)

	cond2 = np.where( (is_PtDens_pos == True) & (is_PtPt_pos == True),  True, False)

	# # condition 3
	# # read redox label
	redox_file = "{0}/redox/{1}/{2}_{3}___{4}.txt".format(myinput_dir,
		task, fix_val, final_state, init_state)
	redox_label = np.loadtxt(redox_file)
	cond3 = redox_label

	conds_vals = list(itertools.product((1, 0, -1), (True, False)))
	# # # showing result of manual dtree
	lbls = set(redox_label.ravel())

	data = []
	for lbl in lbls:

		for conds_val in conds_vals:
			# # v1 include 1, 0, -1; v2 include True False only
			v1, v2 = conds_val
			concern_region = np.where( (cond1 == v1) & (cond2 == v2) & (redox_label == lbl),  redox_label, np.nan)
			save_at = "{0}/mechanism/{1}/{2}_{6}/cond1_{3}/cond2_{4}/redox_{5}.pdf".format(result_dir, task, fix_val, v1, v2, lbl, "___".join(diff_state))
			
			n_valid = (concern_region.size - np.count_nonzero(np.isnan(concern_region))) / float(total_px)
			
			# # save to csv
			row = dict({"task":task, "fix_state":fix_val, "diff_range":"___".join(diff_state), 
						"cond1":v1, "cond2":v2, "redox_state":lbl, "ncount":n_valid})

			data.append(row)


			plot_density(values=concern_region, save_at=save_at,  # diff_PtDens_lbl
				cmap_name="jet", vmin=0, vmax=8, is_save2input=True)

			# # plot joint with dmin
			# dmin_file = "{0}/task4/dmin{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
			# dmin_value = np.loadtxt(dmin_file)
			# dmin_value_concern = np.where( (cond1 == v1) & (cond2 == v2) & (redox_label == lbl),  dmin_value, np.nan)

			# save_at = "{0}/mechanism/{1}/{2}_{6}/cond1_{3}/cond2_{4}/dmin_of_redox_{5}.pdf".format(result_dir, task, fix_val, v1, v2, lbl, "___".join(diff_state))
			# # print (dmin_value_concern)
			# x = dmin_value_concern[~np.isnan(dmin_value_concern)]

			# if len(x) != 0:
			# 	plot_hist(x, save_at=save_at, label=save_at.replace(result_dir, ""), nbins=50)

			# joint_plot(x=dmin_value_concern.ravel(), y=concern_region.ravel(), 
			# 	xlabel="dmin_{0}{1}{2}".format(fixP, fixT, fixV), ylabel="redox_state", 
			# 	xlim=[-2, 40], ylim=None,
			# 	title=save_at.replace(result_dir, ""),
			# 	save_at=save_at)
			
	summary_df = pd.DataFrame(data)
	save_at = "{0}/mechanism/{1}/{2}_{3}.csv".format(result_dir, task, fix_val, "___".join(diff_state))
	summary_df.to_csv(save_at)
	for conds_val in conds_vals:
		# # v1 include 1, 0, -1; v2 include True False only
		v1, v2 = conds_val
		concern_region = np.where((cond1 == v1) & (cond2 == v2), cond3, np.nan)
		# concern_region = cond3[idx]
		save_at = "{0}/mechanism/{1}/{2}_{5}/cond1_{3}/cond2_{4}/all_redox_states.pdf".format(result_dir, task, fix_val, v1, v2, "___".join(diff_state))
		plot_density(values=concern_region, save_at=save_at,  # diff_PtDens_lbl
			cmap_name="jet", vmin=0, vmax=8, is_save2input=True)
	print (redox_label)


if __name__ == "__main__":



	tasks = ["diff_t", "diff_v", "diff_p"] #  "diff_t", "diff_v", "diff_p"
	

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
				mechanism(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_P, task=task)
			
			if task == "diff_v":
				fixT, fixP, diff_V = comb
				mechanism(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_V, task=task)
	
			if task == "diff_t":
				diff_T, fixP, fixV = comb
				mechanism(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_T, task=task)

		# 	break
		# break




