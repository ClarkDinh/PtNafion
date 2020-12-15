import os, pickle, glob, ntpath
import numpy as np
import pandas as pd
from tensorflow.io import gfile
from plot import plot_density
SIZE_X = 1024 
SIZE_Y = 1024 

# # for nguyen@macpro
MainDir = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs" 

# # for nguyen@trump-gw
# MainDir =  "/home/nguyen/work/Nagoya_cfxafs" 
job = "Whole/fresh_CT13/bkg"
text_dir = "/text/"
fig_dir = "/fig/"
SIZE_Z = dict({"bkg": 546})

ResultDir = "{}/result".format(MainDir)
InputDir = "{}/input".format(MainDir)


MinBkg, MaxBkg = 0.0, 0.05 # max(all_max): 0.2991335690021515
threshold = 0.05/20.0
is_plot = False
index2remove = range(0, 30) # # for crop image of redox, dmin to redox

# norm_value_range = { 
# 	"Pt-density": (min_Ptdens, max_Ptdens), # (0.0, 0.0041548326), 
# 	"Pt-valence": (min_Ptval, max_Ptval), # (0.0, 1.219965), 
# 	"Pt-Pt": (min_PtPt, max_PtPt), # (0.0, 11.999759), 
# 	"Pt-O": (min_PtO, max_PtO), # (0.0, 1.999747)
# 	"morphology": (min_morph, max_morph), # (0.0, 0.0037971088), 
# }

def get_vmin_vmax_diff(feature):
	lb, ub = norm_value_range[feature]
	vmin, vmax = -ub, ub
	return (vmin, vmax)

def get_basename(filename):
    head, tail = ntpath.split(filename)
    basename = os.path.splitext(tail)[0]
    return tail

def get_org_dir(p, feature, t, v, ftype="txt"):
	orgdir = "{0}/{1}/{2}/{3}{4}_{2}.{5}".format(myinput_dir, 
		p, feature, t, v, ftype)
	return orgdir

def get_dmin(fixP, fixT, fixV):
	# # with bkg def 0.0008 use task4
	# # with bkg def 0.0003 use task4_0928_bkg_revise
    dmin_file = "{0}/feature/task4_0928_bkg_revise/dmin_{1}{2}{3}_morphology.txt".format(input_dir, fixP, fixT, fixV)
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


def save_layer2text(data, file):
	makedirs(file)
	with open(file,"w") as handler:
		data = np.savetxt(fname=handler, X=data)

def load_1layer(file):
	data = np.loadtxt(file)
	return data


def get_subdirs(sdir):
	subdirs = glob.glob(sdir+"/*")
	return subdirs

def cat2Dto3D(jobdir, init_layer, final_layer):
	layers = []
	all_min = []
	all_max = []

	if (init_layer is not None) and (final_layer is not None):
		files = [InputDir+"/"+jobdir+"/r"+"{0:04}".format(k)+".txt" for k in range(init_layer, final_layer+1)]
	else:
		files = get_subdirs(InputDir+"/"+jobdir)
	for f in files:
		this_layer = load_1layer(file=f)
		this_min = np.min(this_layer)
		this_max = np.max(this_layer)

		all_min.append(this_min)
		all_max.append(this_max)
		layers.append(this_layer)
	layers = np.array(layers)

	return layers, all_min, all_max


def txt2png(jobdir, init_layer, final_layer):
	bkg_thresh = MaxBkg / 20

	if (init_layer is not None) and (final_layer is not None):
		files = [InputDir+"/txt/"+jobdir+"/r"+"{0:04}".format(k)+".txt" for k in range(init_layer, final_layer+1)]
	else:
		print(InputDir+"/txt/"+jobdir)
		files = get_subdirs(InputDir+"/txt/"+jobdir)
		print(files)

	for f in files:
		basename = get_basename(filename=f)
		basename = basename[:basename.find(".")]
		this_layer = load_1layer(file=f)
		save_at = InputDir+"/label/"+jobdir+"/"+basename+".png"
		X_lbl = np.where(this_layer > bkg_thresh, 1.0, 0.0)
		print (X_lbl)
		plot_density(values=X_lbl, save_at=save_at,
			cmap_name="Greys", 
			title=None, vmin=None, vmax=None, is_save2input=None)

		save_at = InputDir+"/label_txt/"+jobdir+"/"+basename+".txt"
		
		save_layer2text(X_lbl, file=save_at)


def makedirs(file):
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))

def load_pickle(filename):
  if not gfile.exists(filename):
    raise NameError("ERROR: the following data not available \n" + filename)
  data = pickle.load(gfile.GFile(filename, "rb"))
  return data

def dump_pickle(data, filename):
	pickle.dump(data, gfile.GFile(filename, 'wb'))

def get_prefix(kwargs):
	# "min_cluster_size":min_cluster_size, 
	# 	"min_samples":min_samples, 
	# 	"alpha":alpha, "allow_single_cluster":allow
	keys_list = ["min_cluster_size", "min_samples", 
					"alpha", "allow_single_cluster"]
	prefix = "/params"
	for p in keys_list:
		assert p in kwargs.keys()

		prefix += "{}-".format(kwargs[p])

	return prefix

if __name__ == "__main__":
	tasks = ["bkg", "a0", "b0", "c0"]
	main_jobdir = "Whole/fresh_CT13"
	# txt2png(jobdir=jobdir, init_layer=0, final_layer=120)
	for task in tasks:
		jobdir = main_jobdir+"/"+task
		txt2png(jobdir=jobdir, init_layer=None, final_layer=None)




