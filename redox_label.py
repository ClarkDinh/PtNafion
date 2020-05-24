import numpy as np
import matplotlib.pyplot as plt
import time, gc, os, itertools
import pandas as pd

from plot import plot_density, plot_hist, makedirs, joint_plot
from Nafion_constant import *
from pylab import *

def redox_diff_t(fixP, fixV, diff_state, task="diff_t"):
	vmin=1
	vmax=8 
	init_state, final_state = diff_state
	fix_val = "{0}{1}".format(fixP, fixV)


	consider_Ft = "Pt-density"
	prefix_input = fix_val + consider_Ft

	diff_PtDens = "{0}/task1/{1}/{2}_{3}____{4}.txt".format(input_dir, 
					task, prefix_input, 
					
					init_state, final_state, task)
	diff_PtDens_val, is_diff_PtDens_pos = pos_neg_lbl_cvt(inputfile=diff_PtDens)


	consider_Ft = "Pt-valence"
	prefix_input = fix_val + consider_Ft

	diff_PtVal = "{0}/task1/{1}/{2}_{3}____{4}.txt".format(input_dir, 
					task, prefix_input, 
					
					init_state, final_state, task)
	diff_PtVal_val, is_diff_PtVal_pos = pos_neg_lbl_cvt(inputfile=diff_PtVal)


	consider_Ft = "Pt-Pt"
	prefix_input = fix_val + consider_Ft
	
	diff_PtPt = "{0}/task1/{1}/{2}_{3}____{4}.txt".format(input_dir, 
					task, prefix_input, 
					
					init_state, final_state, task)
	diff_PtPt_val, is_diff_PtPt_pos = pos_neg_lbl_cvt(inputfile=diff_PtPt)

	# # all shape checked
	# print ("is_diff_PtDens_pos", is_diff_PtDens_pos)
	# print ("is_diff_PtVal_pos", is_diff_PtVal_pos)
	# print ("is_diff_PtPt_pos", is_diff_PtPt_pos)

	# # for binary label
	# redox_state = np.where(
	# 	((is_diff_PtDens_pos == True) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == True)),  
	# 	1, -1)

	# # for octary label
	redox_states = redox_state_lbl(is_diff_PtDens_pos=is_diff_PtDens_pos, 
		is_diff_PtVal_pos=is_diff_PtVal_pos, 
		is_diff_PtPt_pos=is_diff_PtPt_pos)

	redox_sum = sum(redox_states, axis=0)

	# # to plot

	cmap_name="jet"
	prefix = "{0}/redox/{1}/{1}_{2}{3}_redox_{4}____{5}".format(result_dir,
		task, fixP, fixV, init_state, final_state) 

	save_at = prefix + ".pdf"
	plot_density(values=redox_sum, save_at=save_at,  # diff_PtDens_lbl
		cmap_name=cmap_name, vmin=vmin, vmax=vmax)

	# # save to txt
	save_txt = prefix.replace("result", "input") + ".txt"
	makedirs(save_txt)

	np.savetxt(save_txt, redox_sum)

	for i, rst in enumerate(redox_states):
		tmp_save_at = "{0}/redox_{1}.pdf".format(prefix, i+1)
		plot_density(values=rst, save_at=tmp_save_at,  
			cmap_name=cmap_name, vmin=vmin, vmax=vmax)



def redox_diff_v(fixT, fixP, diff_state, task="diff_v"):

	init_state, final_state = diff_state # # here is diff voltage
	fix_val = "{0}{1}".format(fixT, fixP)
	
	consider_Ft = "Pt-density"
	prefix_input = fix_val + consider_Ft
	diff_PtDens = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					init_state, final_state)
	diff_PtDens_val, is_diff_PtDens_pos = pos_neg_lbl_cvt(inputfile=diff_PtDens)



	consider_Ft = "Pt-valence"
	prefix_input = fix_val + consider_Ft
	diff_PtVal = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					init_state, final_state)
	diff_PtVal_val, is_diff_PtVal_pos = pos_neg_lbl_cvt(inputfile=diff_PtVal)



	consider_Ft = "Pt-Pt"
	prefix_input = fix_val + consider_Ft
	diff_PtPt = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					init_state, final_state)
	diff_PtPt_val, is_diff_PtPt_pos = pos_neg_lbl_cvt(inputfile=diff_PtPt)

	# all shape checked
	# print ("is_diff_PtDens_pos", diff_PtDens_val)
	# print ("is_diff_PtVal_pos", diff_PtVal_val)
	# print ("is_diff_PtPt_pos", diff_PtPt_val)


	redox_states = redox_state_lbl(is_diff_PtDens_pos=is_diff_PtDens_pos, 
		is_diff_PtVal_pos=is_diff_PtVal_pos, 
		is_diff_PtPt_pos=is_diff_PtPt_pos)


	redox_sum = sum(redox_states, axis=0)

	# # to plot
	vmin=1
	vmax=8 
	cmap_name="jet"
	# cmap_name = cm.get_cmap('PiYG', 8)

	print (redox_sum)

	prefix = "{0}/redox/{1}/{2}_{3}___{4}".format(result_dir,
		task, fix_val,  init_state, final_state) 

	save_at = prefix + ".pdf"
	plot_density(values=redox_sum, save_at=save_at,  # diff_PtDens_lbl
		cmap_name=cmap_name, vmin=vmin, vmax=vmax)

	# # save to txt
	save_txt = prefix.replace("result", "input") + ".txt"
	makedirs(save_txt)

	np.savetxt(save_txt, redox_sum)

	for i, rst in enumerate(redox_states):
		tmp_save_at = "{0}/redox_{1}.pdf".format(prefix, i+1)
		plot_density(values=rst, save_at=tmp_save_at,  
			cmap_name=cmap_name, vmin=vmin, vmax=vmax)



def redox_lbl(fixT, fixP, fixV, diff_state, task="diff_p"):
	# ADT5k1VPt-density_CCM-Nafion____CCMcenter.pdf
	init_state, final_state = diff_state # # here is diff voltage

	if task == "diff_p":
		fix_val = "{0}{1}".format(fixT, fixV)
	if task == "diff_v":
		fix_val = "{0}{1}".format(fixT, fixP)
	if task == "diff_t":
		fix_val = "{0}{1}".format(fixP, fixV)
	
	consider_Ft = "Pt-density"
	prefix_input = fix_val + consider_Ft
	diff_PtDens = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					init_state, final_state)
	diff_PtDens_val, is_diff_PtDens_pos = pos_neg_lbl_cvt(inputfile=diff_PtDens)



	consider_Ft = "Pt-valence"
	prefix_input = fix_val + consider_Ft
	diff_PtVal = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					init_state, final_state)
	diff_PtVal_val, is_diff_PtVal_pos = pos_neg_lbl_cvt(inputfile=diff_PtVal)



	consider_Ft = "Pt-Pt"
	prefix_input = fix_val + consider_Ft
	diff_PtPt = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					init_state, final_state)
	diff_PtPt_val, is_diff_PtPt_pos = pos_neg_lbl_cvt(inputfile=diff_PtPt)

	# all shape checked
	# print ("is_diff_PtDens_pos", diff_PtDens_val)
	# print ("is_diff_PtVal_pos", diff_PtVal_val)
	# print ("is_diff_PtPt_pos", diff_PtPt_val)


	redox_states = redox_state_lbl(is_diff_PtDens_pos=is_diff_PtDens_pos, 
		is_diff_PtVal_pos=is_diff_PtVal_pos, 
		is_diff_PtPt_pos=is_diff_PtPt_pos)


	redox_sum = sum(redox_states, axis=0)

	# # to plot
	vmin=1
	vmax=8 
	cmap_name="jet"
	# cmap_name = cm.get_cmap('PiYG', 8)

	print (redox_sum)

	prefix = "{0}/redox/{1}/{2}_{3}___{4}".format(result_dir,
		task, fix_val,  init_state, final_state) 

	save_at = prefix + ".pdf"
	plot_density(values=redox_sum, save_at=save_at,  # diff_PtDens_lbl
		cmap_name=cmap_name, vmin=vmin, vmax=vmax)

	# # save to txt
	save_txt = prefix.replace("result", "input") + ".txt"
	makedirs(save_txt)

	np.savetxt(save_txt, redox_sum)

	for i, rst in enumerate(redox_states):
		tmp_save_at = "{0}/redox_{1}.pdf".format(prefix, i+1)
		plot_density(values=rst, save_at=tmp_save_at,  
			cmap_name=cmap_name, vmin=vmin, vmax=vmax)



def redox_state_lbl(is_diff_PtDens_pos, is_diff_PtVal_pos, is_diff_PtPt_pos):
	redox_state_8 = np.where(
		((is_diff_PtDens_pos == True) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == True)),  
		8, 0)

	redox_state_7 = np.where(
		((is_diff_PtDens_pos == True) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == False)),  
		7, 0)

	redox_state_6 = np.where(
		((is_diff_PtDens_pos == True) & (is_diff_PtVal_pos == False) & (is_diff_PtPt_pos == True)),  
		6, 0)

	redox_state_5 = np.where(
		((is_diff_PtDens_pos == True) & (is_diff_PtVal_pos == False) & (is_diff_PtPt_pos == False)),  
		5, 0)

	redox_state_4 = np.where(
		((is_diff_PtDens_pos == False) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == True)),  
		4, 0)

	redox_state_3 = np.where(
		((is_diff_PtDens_pos == False) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == False)),  
		3, 0)

	redox_state_2 = np.where(
		((is_diff_PtDens_pos == False) & (is_diff_PtVal_pos == False) & (is_diff_PtPt_pos == True)),  
		2, 0)

	redox_state_1 = np.where(
		((is_diff_PtDens_pos == False) & (is_diff_PtVal_pos == False) & (is_diff_PtPt_pos == False)),  
		1, 0)
	redox_states = [redox_state_1, redox_state_2, redox_state_3, redox_state_4,
				redox_state_5, redox_state_6, redox_state_7, redox_state_8]

	return redox_states






def join_redox_dmin(fixT, fixP, fixV, diff_state, save_at, task="diff_p"):
	init_state, final_state = diff_state # # here is diff voltage

# dminCCM-NafionADT15k04V_morphology.txt
# dminCCM-NafionFresh04V_morphology.txt

	# # get dmin
	dmin_file = "{0}/task4/dmin{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
	dmin_value = np.loadtxt(dmin_file)

	save_at = "{0}/dmin_redox/{1}/{2}_{3}___{4}_redox.pdf".format(result_dir,
		task, fix_val,  init_state, final_state) 

	joint_plot(x=dmin_value.ravel(), y=redox_value.ravel(), 
		xlabel="dmin_{0}{1}{2}".format(fixP, fixT, fixV), ylabel="redox_state", 
		xlim=[-2, 40], ylim=None,
		title=save_at.replace(result_dir, ""),
		save_at=save_at)


def save_diff_2csv(df, fixT, fixP, fixV, diff_state, save_at, task="diff_p"):
	init_state, final_state = diff_state # # here is diff voltage

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
		# diff_val_file = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
		# 			task, prefix_input, 
		# 			init_state, final_state)

		diff_val_file = "{0}/task1/{1}/{2}_{3}___{4}.txt".format(input_dir, 
					task, prefix_input, 
					init_state, final_state)
		
		diff_val, is_diff_val_pos = pos_neg_lbl_cvt(inputfile=diff_val_file)

		feature_name = diff_val_file.replace(input_dir, "")
		try:
			df[feature_name] = diff_val.ravel()
		except Exception as e:
			failed_feature.append(diff_val_file)
			pass


	# # save redox value
	redox_file = "{0}/redox/{1}/{2}_{3}___{4}.txt".format(myinput_dir,
		task, fix_val,  init_state, final_state) 
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


def pos_neg_lbl_cvt(inputfile):
	try:
		data = np.loadtxt(inputfile)
		
	except Exception as e:
		revise_file = inputfile.replace("___", "____")
		data = np.loadtxt(revise_file)
		if not os.path.isfile(revise_file):
			print ("TOTALLY WRONG DIR HERE!", inputfile)
		pass

	# # get label under conditions
	lbl = np.where(data>=0, True, False)
	# save_at = "{0}/tmp.pdf".format(result_dir) # for test
	# plot_density(values=diff_PtDens_val, save_at=save_at,  # diff_PtDens_lbl
	# 	cmap_name="bwr", vmin=None, vmax=None)

	return data, lbl



if __name__ == "__main__":
	maindir = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion"
	input_dir = "{}/fromQuan/v3_0520/feature".format(maindir)
	result_dir = "{}/result".format(maindir)
	myinput_dir = "{}/input".format(maindir)

	# # in considering diff_t
	tasks = ["diff_p", "diff_t", "diff_v"]
	is_redox_lbl =False
	is_redox2dmin = False
	is_save2csv = True

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
			redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_V, task=task)
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
					redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_P, task=task)

				
				if task == "diff_v":
					fixT, fixP, diff_V = comb
					redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_V, task=task)
		
				if task == "diff_t":
					diff_T, fixP, fixV = comb
					redox_lbl(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_T, task=task)


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
					fixP = diff_P[0]
					join_redox_dmin(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_P, task=task)

				
				if task == "diff_v":
					fixT, fixP, diff_V = comb
					fixV = diff_V[0]
					join_redox_dmin(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_V, task=task)
		
				if task == "diff_t":
					diff_T, fixP, fixV = comb
					fixT = diff_T[0]
					join_redox_dmin(fixT=fixT, fixP=fixP, fixV=fixV, diff_state=diff_T, task=task)

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
					df, failed_feature = save_diff_2csv(fixT=fixT, fixP=fixP, fixV=fixV, 
						diff_state=diff_V, save_at=save_at, task=task)
		
				if task == "diff_t":
					diff_T, fixP, fixV = comb
					fixT = diff_T[0]
					df, failed_feature = join_redox_dmin(fixT=fixT, fixP=fixP, fixV=fixV, 
						diff_state=diff_T, save_at=save_at, task=task)
				print ("failed_feature", failed_feature)
				all_failed_feature.append(failed_feature)

			# 	break
			# break
	print (all_failed_feature)

























