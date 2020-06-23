


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


maindir = "/Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_Nafion"
input_dir = "{}/fromQuan/v3_0520/feature".format(maindir)
result_dir = "{}/result".format(maindir)
myinput_dir = "{}/input".format(maindir)


def get_org_dir(p, feature, t, v, ftype="txt"):
	orgdir = "{0}/{1}/{2}/{3}{4}_{2}.{5}".format(myinput_dir, 
		p, feature, t, v, ftype)
	return orgdir