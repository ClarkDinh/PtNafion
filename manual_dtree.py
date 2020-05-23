import numpy as np
import matplotlib.pyplot as plt
import time, gc, os, itertools
import pandas as pd

from plot import plot_density, plot_hist, makedirs
from Nafion_constant import *
from pylab import *
from redox_label import pos_neg_lbl_cvt



def mechanism(fixT, fixP, fixV, diff_state, task):
	init_state, final_state = diff_state 

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
					init_state, final_state)
	diff_PtDens_val, is_diff_PtDens_pos = pos_neg_lbl_cvt(inputfile=diff_PtDens)
	cond1 = is_diff_PtDens_pos

	# # condition 2
	# # Pt Dens and PtPt

	PtDens = get_org_dir(p=fixP, feature="Pt-density", t=fixT, v=fixV, ftype="txt")
	PtDens_val, is_PtDens_pos = pos_neg_lbl_cvt(inputfile=PtDens)


	PtPt = get_org_dir(p=fixP, feature="Pt-Pt", t=fixT, v=fixV, ftype="txt")
	PtPt_val, is_PtPt_pos = pos_neg_lbl_cvt(inputfile=PtPt)

	cond2 = np.where( (is_PtDens_pos == True) & (is_PtPt_pos == True),  True, False)

	# # condition 3
	# # read redox label
	redox_file = "{0}/redox/{1}/{2}_{3}___{4}.txt".format(myinput_dir,
		task, fix_val,  init_state, final_state)
	redox_label = np.loadtxt(redox_file)
	cond3 = redox_label

	conds_vals = list(itertools.product((True, False), (True, False)))
	# # # showing result of manual dtree
	lbls = set(redox_label.ravel())
	for lbl in lbls:

		for conds_val in conds_vals:
			v1, v2 = conds_val
			concern_region = np.where( (cond1 == v1) & (cond2 == v2) & (redox_label == lbl),  lbl, np.nan)
			save_at = "{0}/mechanism/cond1_{1}/cond2_{2}/redox_{3}.pdf".format(result_dir, v1, v2, lbl)
			plot_density(values=concern_region, save_at=save_at,  # diff_PtDens_lbl
				cmap_name="jet", vmin=0, vmax=8, is_save2input=True)

		
		

	print (redox_label)


if __name__ == "__main__":
	maindir = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion"
	input_dir = "{}/fromQuan/v3_0520/feature".format(maindir)
	result_dir = "{}/result".format(maindir)
	myinput_dir = "{}/input".format(maindir)


	tasks = ["diff_p"] #  "diff_t", "diff_v"
	

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




