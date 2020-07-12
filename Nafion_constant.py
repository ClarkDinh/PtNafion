


size_x = 512
size_y = 512 

Positions = ["CCM-Nafion", "CCMcenter"]
Times = ["Fresh", "ADT5k", "ADT15k"]
Voltages = ["04V", "1V"]

Features = ["morphology", "Pt-density", "Pt-valence", "Pt-O", "Pt-Pt"]


dP = [["CCM-Nafion", "CCMcenter"]]
dT = [["Fresh", "ADT5k"], ["Fresh", "ADT15k"], ["ADT5k", "ADT15k"]]
dT_tunes = [["Fresh", "ADT15k"]]

dV = [["04V", "1V"]]
dV_tunes = [["1V"]]



maindir = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion"
input_dir = "{}/fromQuan/v3_0520/feature".format(maindir)
result_dir = "{}/result".format(maindir)
myinput_dir = "{}/input".format(maindir)


min_Ptdens, max_Ptdens = 0.0, 0.002# 0.005
min_Ptval, max_Ptval = 0.0, 1.25 #
min_PtPt, max_PtPt = 0.0, 12 # 
min_morph, max_morph = 0.0, 0.002 # 0.025
min_PtO, max_PtO = 0.0, 0.3 # 1.5

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