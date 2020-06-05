


import os
import itertools
import numpy as np

cwd = os.getcwd()

y_size = 512 # y_size: number of rows
x_size = 512 # x_size: number of cols
z_size = 1

row_indexes = ["y{}".format(k) for k in range(y_size)]
col_indexes = ["x{}".format(k) for k in range(x_size)]



Pt_dens_range = (0.0001, 0.0006)
Co_dens_range = (0.00004, 0.0003)


input_dir = "{0}/../input".format(cwd)
result_dir = "{0}/../result".format(cwd)



catgory_folders = [ "Co_density", "Co_Pt_ratio", "delta density",
				 "morphology", "Pt_density", "Pt_valence"]



def get_subfolder(catgor_folder):
	sub_cats = dict({
		"Co_density": ["e_Co_fresh", "e_Co_21k", "e_Co_34k"],
		"Pt_density": ["e_Pt_fresh", "e_Pt_21k", "e_Pt_34k"],
		"morphology": ["fresh", "21k", "34k"],
		"Pt_valence": ["v_04_fresh", "v_04_21k", "v_04_34k", "v_10_fresh", "v_10_21k", "v_10_34k"],
		"delta density": ["d_Co", "d_Pt"]

		})

	if catgor_folder in sub_cats:
		return sub_cats[catgor_folder]
	else:
		d = "{0}/input_raw/{1}".format(input_dir, catgor_folder)
		sub_folders = [o for o in os.listdir(d) 
						if os.path.isdir(os.path.join(d,o))]
		return sub_folders


if __name__ == "__main__":

	sub_cats = get_subfolder(catgor_folder="delta density")
	print (sub_cats)


