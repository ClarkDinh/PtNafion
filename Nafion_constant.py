import os
import numpy as np
import pandas as pd

size_x = 512
size_y = 512 

Positions = ["CCM-Nafion", "CCMcenter"]
Times = ["Fresh", "ADT15k"] # ADT5k
Voltages = ["04V", "1V"] # 

Features = ["morphology", "Pt-density", "Pt-valence", "Pt-O", "Pt-Pt"]


dP = [["CCM-Nafion", "CCMcenter"]]
# dT = [["Fresh", "ADT5k"], ["ADT15k", "Fresh"], ["ADT5k", "ADT15k"]]
dT_tunes = [["ADT15k", "Fresh"]]
dT = [["ADT15k", "Fresh"]]

dV = [["1V", "04V"]]
dV_tunes = [["1V", "04V"]]



maindir = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion"
# input_dir = "{}/fromQuan/v3_0520/feature".format(maindir)
# input_dir = "{}/fromQuan/v4_0526/feature".format(maindir)
result_dir = "{}/result".format(maindir)
myinput_dir = "{}/input".format(maindir)
input_dir = "{}/input/2020-06-17-NewRequest/txt".format(maindir)


min_Ptdens, max_Ptdens = 0.0, 0.002# 0.005
min_Ptval, max_Ptval = 0.0, 1.25 #
min_PtPt, max_PtPt = 0.0, 12.0 # 
min_morph, max_morph = 0.0, 0.002 # 0.025
min_PtO, max_PtO = 0.0, 0.3 # 1.5

index2remove = range(0, 30) # # for crop image of redox, dmin to redox

norm_value_range = { 
	"Pt-density": (min_Ptdens, max_Ptdens), # (0.0, 0.0041548326), 
	"Pt-valence": (min_Ptval, max_Ptval), # (0.0, 1.219965), 
	"Pt-Pt": (min_PtPt, max_PtPt), # (0.0, 11.999759), 
	"Pt-O": (min_PtO, max_PtO), # (0.0, 1.999747)
	"morphology": (min_morph, max_morph), # (0.0, 0.0037971088), 
}

def get_vmin_vmax_diff(feature):
	lb, ub = norm_value_range[feature]
	vmin, vmax = -ub, ub
	return (vmin, vmax)

def get_org_dir(p, feature, t, v, ftype="txt"):
	orgdir = "{0}/{1}/{2}/{3}{4}_{2}.{5}".format(myinput_dir, 
		p, feature, t, v, ftype)
	return orgdir

def get_dmin(fixP, fixT, fixV):
    dmin_file = "{0}/feature/task4/dmin_{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
    dmin_value = np.loadtxt(dmin_file)
    dmin_value = np.delete(dmin_value, index2remove, 1)  # delete second column of C
    return dmin_value.ravel()

def remove_nan(ignore_first,matrix1,matrix2,lbl1,lbl2):
    if ignore_first is not None:
        matrix1 = np.delete(matrix1,ignore_first)
        matrix2 = np.delete(matrix2,ignore_first)


    nan_of_matrix1 = np.argwhere(np.isnan(matrix1))
    nan_of_matrix2 = np.argwhere(np.isnan(matrix2))
    print("before:", len(nan_of_matrix2))

    # only for task 1, 2
    more = np.argwhere(matrix2>100)
    nan_of_matrix2 = np.concatenate((more, nan_of_matrix2))


    more = np.argwhere(matrix2<-100)
    nan_of_matrix2 = np.concatenate((more, nan_of_matrix2))

    print("after:", len(nan_of_matrix2))
    # print("Check nan here:", nan_of_matrix1, nan_of_matrix2, len(nan_of_matrix2), len(matrix2))
    matrix1 = np.delete(matrix1,np.concatenate((nan_of_matrix1,nan_of_matrix2),axis=0))
    matrix2 = np.delete(matrix2,np.concatenate((nan_of_matrix1,nan_of_matrix2),axis=0))
    # print("value matrix2:", matrix2)
    df = pd.DataFrame(columns=[lbl1,lbl2])
    df[lbl1] = matrix1
    df[lbl2] = matrix2
    return df

def pos_neg_lbl_cvt(inputfile, is_get_zero=False):
	try:
		data = np.loadtxt(inputfile)
		
	except Exception as e:
		revise_file = inputfile.replace("___", "____")
		data = np.loadtxt(revise_file)
		if not os.path.isfile(revise_file):
			print ("TOTALLY WRONG DIR HERE!", inputfile)
		pass

	# # get label under conditions
	if is_get_zero:
		# lbl_zero = np.where(data==0, 0, False)
		lbl_pos = np.where(data>0, 1, 0)
		lbl_neg = np.where(data<0, -1, 0)
		lbl = lbl_neg + lbl_pos
		
	else:
		lbl = np.where(data>=0, True, False)

	# save_at = "{0}/tmp.pdf".format(result_dir) # for test
	# plot_density(values=diff_PtDens_val, save_at=save_at,  # diff_PtDens_lbl
	# 	cmap_name="bwr", vmin=None, vmax=None)
	return data, lbl

def redox_state_lbl(is_diff_PtO_pos, is_diff_PtVal_pos, is_diff_PtPt_pos):
	redox_state_8 = np.where(
		((is_diff_PtO_pos == True) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == True)),  
		8.0, 0.0)

	redox_state_7 = np.where(
		((is_diff_PtO_pos == True) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == False)),  
		7.0, 0.0)

	redox_state_6 = np.where(
		((is_diff_PtO_pos == True) & (is_diff_PtVal_pos == False) & (is_diff_PtPt_pos == True)),  
		6.0, 0.0)

	redox_state_5 = np.where(
		((is_diff_PtO_pos == True) & (is_diff_PtVal_pos == False) & (is_diff_PtPt_pos == False)),  
		5.0, 0.0)

	redox_state_4 = np.where(
		((is_diff_PtO_pos == False) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == True)),  
		4.0, 0.0)

	redox_state_3 = np.where(
		((is_diff_PtO_pos == False) & (is_diff_PtVal_pos == True) & (is_diff_PtPt_pos == False)),  
		3.0, 0.0)

	redox_state_2 = np.where(
		((is_diff_PtO_pos == False) & (is_diff_PtVal_pos == False) & (is_diff_PtPt_pos == True)),  
		2.0, 0.0)

	redox_state_1 = np.where(
		((is_diff_PtO_pos == False) & (is_diff_PtVal_pos == False) & (is_diff_PtPt_pos == False)),  
		1.0, 0.0)
	redox_states = [redox_state_1, redox_state_2, redox_state_3, redox_state_4,
				redox_state_5, redox_state_6, redox_state_7, redox_state_8]

	return redox_states
