import cv2, glob, os, copy, ntpath
import matplotlib.pyplot as plt
import numpy as np
def makedirs(file):
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))

def get_basename(filename):
    head, tail = ntpath.split(filename)
    basename = os.path.splitext(tail)[0]
    return tail

position = "CCM-Nafion" # CCM-Nafion, CCMcenter
folder="/media/nguyen/work/Nagoya_Nafion/input/2020-06-17-NewRequest/tif/"+position 
# all_tif_imgs = glob.glob('{}/*1V_Pt-Pt.tif'.format(folder))
all_tif_imgs = glob.glob('{}/*.tif'.format(folder))


minmax_dict = dict({})

min_Ptdens, max_Ptdens = 0.0, 0.002# 0.005
min_Ptval, max_Ptval = 0.0, 1.25 #
min_PtPt, max_PtPt = 0.0, 12 # 
min_morph, max_morph = 0.0, 0.002 # 0.025
min_PtO, max_PtO = 0.0, 0.3 # 1.5

save_cfg = { 
'CCM-Nafion/ADT15k04V_Pt-density': (min_Ptdens, max_Ptdens), # (0.0, 0.0041548326), 
'CCM-Nafion/ADT15k1V_Pt-density': (min_Ptdens, max_Ptdens), # (0.0, 0.002392245),
'CCM-Nafion/ADT15k04V_Pt-valence': (min_Ptval, max_Ptval), # (0.0, 1.219965), 
'CCM-Nafion/ADT15k1V_Pt-valence': (min_Ptval, max_Ptval), # (0.0, 1.2197714),
'CCM-Nafion/ADT15k04V_Pt-Pt': (min_PtPt, max_PtPt), # (0.0, 11.999759), 
'CCM-Nafion/ADT15k1V_Pt-Pt': (min_PtPt, max_PtPt), # (0.0, 12), # #old (-1012.3431, 11,999)
'CCM-Nafion/ADT15k04V_Pt-O': (min_PtO, max_PtO),  # (0.0, 1.999933)
'CCM-Nafion/ADT15k1V_Pt-O': (min_PtO, max_PtO), # (0.0, 1.999747)
'CCM-Nafion/ADT15k04V_morphology': (min_morph, max_morph), # (0.0, 0.011654467), 
'CCM-Nafion/ADT15k1V_morphology': (min_morph, max_morph), # (0.0, 0.0037971088), 


'CCM-Nafion/Fresh04V_Pt-density': (min_Ptdens, max_Ptdens), # (1.2098759e-27, 0.0047670836), # 
'CCM-Nafion/Fresh1V_Pt-density': (min_Ptdens, max_Ptdens), # (0.0, 0.0036530301),
'CCM-Nafion/Fresh04V_Pt-valence': (min_Ptval, max_Ptval), # (0.100578874, 1.219928),
'CCM-Nafion/Fresh1V_Pt-valence': (min_Ptval, max_Ptval), # (0.0, 1.2199816),
'CCM-Nafion/Fresh04V_Pt-Pt': (min_PtPt, max_PtPt), # (1.1754756, 11.999926), 
'CCM-Nafion/Fresh1V_Pt-Pt': (min_PtPt, max_PtPt), # (-3.5707866e+16, 11.999695), 
'CCM-Nafion/Fresh04V_Pt-O': (min_PtO, max_PtO),  # (1.9341861e-11, 1.9996669)
'CCM-Nafion/Fresh1V_Pt-O': (min_PtO, max_PtO), # (0.0, 1.9999839)
'CCM-Nafion/Fresh04V_morphology': (min_morph, max_morph), # (5.225914e-07, 0.024229323), 
'CCM-Nafion/Fresh1V_morphology': (min_morph, max_morph), #(0.0, 0.00545342), 

'CCMcenter/ADT15k04V_Pt-density': (min_Ptdens, max_Ptdens), # (0.0, 0.0036105984), 
'CCMcenter/ADT15k1V_Pt-density': (min_Ptdens, max_Ptdens), # (0.0, 0.0032233016), 
'CCMcenter/ADT15k04V_Pt-valence': (min_Ptval, max_Ptval), # (0.0, 1.219931), 
'CCMcenter/ADT15k1V_Pt-valence': (min_Ptval, max_Ptval), #(0.0, 1.2199963),
'CCMcenter/ADT15k04V_Pt-Pt': (min_PtPt, max_PtPt), #  (0.0, 11.999855),
'CCMcenter/ADT15k1V_Pt-Pt': (min_PtPt, max_PtPt), #  (0.0, 11.999911), 
'CCMcenter/ADT15k04V_Pt-O': (min_PtO, max_PtO), #  (0.0, 1.9996853), 
'CCMcenter/ADT15k1V_Pt-O': (min_PtO, max_PtO), #  (0.0, 1.9994632), 
'CCMcenter/ADT15k04V_morphology': (min_morph, max_morph), # (0.0, 0.01572743), 
'CCMcenter/ADT15k1V_morphology': (min_morph, max_morph), # (0.0, 0.0043866597), 

'CCMcenter/Fresh04V_Pt-density': (min_Ptdens, max_Ptdens), # (1.3193431e-29, 0.003606605), 
'CCMcenter/Fresh1V_Pt-density': (min_Ptdens, max_Ptdens), # (0.0, 0.003046906), 
'CCMcenter/Fresh04V_Pt-valence': (min_Ptval, max_Ptval), # (0.19646767, 1.2198858), 
'CCMcenter/Fresh1V_Pt-valence': (min_Ptval, max_Ptval), # (0.0, 1.2199731), 
'CCMcenter/Fresh04V_Pt-Pt': (min_PtPt, max_PtPt), # (1.1348895, 11.999994), 
'CCMcenter/Fresh1V_Pt-Pt': (min_PtPt, max_PtPt), # (0.0, 11.99984), 
'CCMcenter/Fresh04V_Pt-O': (min_PtO, max_PtO), # (2.2581765e-11, 1.9996016), 
'CCMcenter/Fresh1V_Pt-O': (min_PtO, max_PtO), # (0.0, 1.9999316), 
'CCMcenter/Fresh04V_morphology': (min_morph, max_morph), # (2.7954906e-07, 0.016310846), 
'CCMcenter/Fresh1V_morphology': (min_morph, max_morph), # (0.0, 0.0052206325), 

}


if __name__ == "__main__":
	for filename in all_tif_imgs:
		img = cv2.imread(filename, -1)
		# print(img)
		saveat=filename.replace("tif", "pdf")
		makedirs(saveat)
		# # 
		basename = position + "/"+ get_basename(filename)

		vmin, vmax = None, None
		# if "1V_Pt-Pt" in str(filename):
		# 	img_cp = copy.copy(img)
		# 	img_cp[img_cp<1e-3] = np.nan
		# 	# vmin, vmax = np.nanmin(img_cp), np.nanmax(img_cp)
		# 	vmin, vmax = np.nanmin(img), np.nanmax(img)
		# 	# plt.imsave(saveat,arr=img, cmap="jet", vmin=vmin, vmax=vmax)
		# 	print("herer:", vmin, vmax)
		# 	print("herer:", filename)
		
		# if "ADT15k1V_morphology" in basename:
		# 	vmin, vmax = 0.0, 0.003

		# if "ADT15k1V_morphology" in basename:
		# 	vmin, vmax = 0.0, 0.003


		vmin, vmax = save_cfg[basename]
		# vmin, vmax = np.nanmin(img), np.nanmax(img)
		# minmax_dict[basename] = (vmin, vmax)

		fig = plt.figure(figsize=(10, 9))
		plt.imshow(img, cmap="jet", vmin=vmin, vmax=vmax)
		plt.title(basename)
		plt.colorbar(shrink=0.8)
		plt.tight_layout()
		# fig.axes.get_xaxis().set_visible(False)
		# fig.axes.get_yaxis().set_visible(False)
		plt.xticks([])
		plt.yticks([])
		plt.savefig(saveat)

		# plt.imsave(saveat,arr=img, cmap="jet")
		savetxt=filename.replace("tif", "txt")
		makedirs(savetxt)
		np.savetxt(fname=savetxt,X=img)
	print(minmax_dict)