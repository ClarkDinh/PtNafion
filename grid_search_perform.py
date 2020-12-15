import sys, copy
from Rubber_constant import *
from itertools import product, combinations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from embedding_space import EmbeddingSpace
from plot import *
from preprocessing import Preprocessing


def get_info(kwargs):
	prefix = get_prefix(kwargs)
	particle_folder = ResultDir + prefix + text_dir + "/particles/" + job+  "/layer_0-546"
	summary_folder = ResultDir + prefix + "/task21"

	slice_3D_folder = ResultDir + prefix + "/slice_3D_bkg"

	summary_file = summary_folder + "/summary.csv"

	summary_file
	if os.path.exists(summary_file):	
		summary_df = pd.read_csv(summary_file, index_col=0)
	else:
		summary_df = []
	return prefix, particle_folder, summary_folder, slice_3D_folder, summary_df

def evaluation():
	min_cluster_size_list = [10, 50, 100, 200, 500]
	min_samples_list = [10, 50, 100, 200, 500]
	allow_single_cluster_list = [True, False] # 
	alpha_list = [ 0.2, 0.5, 0.8 ] # 

	all_kwargs = list(product(min_cluster_size_list, min_samples_list, allow_single_cluster_list, alpha_list))

	grid_search_df = pd.DataFrame(columns=["min_cluster_size", "min_samples", 
				"allow_single_cluster", "alpha", "n_particles"])
	for kw in all_kwargs:
		min_cluster_size, min_samples, allow, alpha = kw[0], kw[1], kw[2], kw[3]
		kwargs = dict(
			{"min_cluster_size":min_cluster_size, 
			"min_samples":min_samples, 
			"alpha":alpha, "allow_single_cluster":allow})
		info = get_info(kwargs=kwargs)
		prefix, particle_folder, summary_folder, slice_3D_folder, summary_df = info
		for k, v in kwargs.items():
			grid_search_df.loc[prefix, k] = v

		grid_search_df.loc[prefix, "n_particles"] = len(summary_df)

	grid_search_df.to_csv(ResultDir+"/grid_search_info.csv")

if __name__ == "__main__":
	reference_folder = InputDir + "/txt/Whole/fresh_CT13"

	# # 1. create_params_grid
	evaluation()

