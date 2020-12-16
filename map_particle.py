import sys, copy
from Rubber_constant import *
from itertools import product, combinations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from embedding_space import EmbeddingSpace
from plot import *
from preprocessing import Preprocessing

def filter_array(x, y):

	zero_y = (y == 0)
	string_y = (y == "All Nan")

	nonvalid_id = zero_y + string_y
	# print (valid_id)
	return x[~nonvalid_id], y[~nonvalid_id]

def normalize(X):
	# # normalize
	scaler = MinMaxScaler().fit(X)
	X_norm = scaler.transform(X)

	return scaler, X_norm

def one_map(df, y_col):
	cols = df.columns
	# # ['bkg_volumes_original', 'bkg_volumes_after', 'bkg_surfaces',
	   # 'bkg_size_z', 'bkg_size_x', 'bkg_size_y', 'bkg_max_value',
	   # 'bkg_sum_value', 'bkg_var_value', 'a_volumes', 'a_max_value',
	   # 'a_sum_value', 'a_var_value', 'b_volumes', 'b_max_value', 'b_sum_value',
	   # 'b_var_value', 'c_volumes', 'c_max_value', 'c_sum_value',
	   # 'c_var_value']

	bkg_props = [['bkg_volumes_after', 'bkg_surfaces', 
			'bkg_size_z', 'bkg_size_x', 'bkg_size_y', 'bkg_max_value', 
			'bkg_sum_value', 'bkg_var_value']]
	bkg_prop_combs = list(combinations(bkg_props, 2))
	print (bkg_prop_combs)

	all_X_cols = [

		[ 'bkg_volumes_after', 'bkg_surfaces'],
		['bkg_size_z', 'bkg_size_x', 'bkg_size_y'], 
		['bkg_max_value', 'bkg_sum_value', 'bkg_var_value'],

		['bkg_volumes_after', 'bkg_surfaces', 'bkg_size_z', 'bkg_size_x', 'bkg_size_y'],
		['bkg_size_z', 'bkg_size_x', 'bkg_size_y', 'bkg_max_value', 'bkg_sum_value', 'bkg_var_value'], 
		['bkg_volumes_after', 'bkg_surfaces', 'bkg_size_z', 'bkg_size_x', 'bkg_size_y', 'bkg_max_value', 'bkg_sum_value', 'bkg_var_value'],

		]

	all_X_cols += bkg_props + bkg_prop_combs

	
	df = df[df[y_col] != "All nan"] # copy.copy()
	print (df[df[y_col] == "All nan"])

	filter_particles = [k for k in df.index if "particle_-1" not in k and "particle_1617" not in k]
	# pd.to_numeric(df[y_col], errors='coerce').isnull().index

	# #
	# abc_props = ["bkg_var_value", "a_volumes", "a_max_value", "a_sum_value"
	# 	"a_var_value", "b_volumes", "b_max_value", "b_sum_value", 
	# 	"b_var_value", "c_volumes", "c_max_value", "c_sum_value", 
	# 	"c_var_value"]
	# filter_particles = df.eq(df.loc[:, 0], axis=0).all(1)
	for X_cols in all_X_cols:
		X = np.array(df.loc[filter_particles, X_cols].values)
		y = np.array(df.loc[filter_particles, y_col].values.ravel())

		print ("X.shape before: ", X.shape)
		X, y = filter_array(X, y)
		print ("X.shape after: ", X.shape)


		# # normalize
		scaler, X_norm = normalize(X)

		n_particles = X_norm.shape[0]
		scale = 300

		mkl_methods = [] # "MLKR", "LFDA"
		dimreduc_methods = ["tsne", "mds", "isomap"]
		methods = mkl_methods + dimreduc_methods

		for method in methods :
			if method in mkl_methods:
				model = EmbeddingSpace(embedding_method=method)
				model.fit(X_train=X_norm, y_train=y)
				X_trans = model.transform(X_val=X_norm, get_min_dist=False)
			
			if method in dimreduc_methods:
				print ("X shape", X.shape)
				print ("y shape", y.shape)

				scaler, Xy_norm = normalize(np.c_[X, y]) # 

				model = Preprocessing(similarity_matrix=Xy_norm)
				
				if method == "tsne":
					X_trans, _ = model.tsne(n_components=2, 
						perplexity=20, 
						early_exaggeration=200, learning_rate=200.0, n_iter=1000,
						n_iter_without_progress=300, min_grad_norm=1e-07, 
						metric='euclidean', init='random',
						verbose=0, random_state=None, method='barnes_hut', 
						angle=0.5, n_jobs=None)
				if method == "isomap":
					X_trans, _ = model.iso_map(
							n_neighbors=int(n_particles / scale), n_components=2, eigen_solver='auto', 
							tol=0, max_iter=None, path_method='auto',
							neighbors_algorithm='auto', n_jobs=None)
				if method == "mds":
					X_trans, _ = model.mds(n_components=2, metric=True, n_init=4, 
						max_iter=300, verbose=0, eps=0.001, n_jobs=None, 
						random_state=None, dissimilarity='euclidean')

		
			save_at = summary_folder + "/{0}/{1}/comb_{2}/{3}/dens.pdf".format(
							y_col, method, len(X_cols), "|".join(X_cols))

			print ("Save at:", save_at)
			title = save_at.replace(ResultDir, "").replace("/", "\n")

			x = X_trans[:, 0]
			y = X_trans[:, 1]
			xlabel = "{0} dim 1".format(method)
			ylabel = "{0} dim 2".format(method)
			joint_plot_2(x=x, y=y, 
				xlabel=xlabel, ylabel=ylabel,
				xlim=(min(x) - 0.1, max(x) + 0.1),
				ylim=(min(y) - 0.1, max(y) + 0.1),
				title=title, save_at=save_at)

			scatter_plot_4(x=x, y=y, color_array=None, xvlines=None, yhlines=None, 
				sigma=None, mode='scatter', lbl=None, name=None, 
				s=30, alphas=0.6, title=title,
				x_label='x', y_label='y', 
				save_file=save_at.replace("dens.pdf", "scatter.pdf"), 
				interpolate=False, color='blue', 
				preset_ax=None, linestyle='-.', marker='o')

			# scatter_plot(x=x, y=y, xvline=None, yhline=None, 
			# 	sigma=None, mode='scatter', 
			# 	lbl=None, #name=filter_particles, 
			# 	x_label=xlabel, y_label=ylabel, 
			# 	save_file=save_at.replace("dens.pdf", "scatter.pdf"), interpolate=False, linestyle='-.',
			# 	color="blue", marker="s"
			# 	)



def map_particles(df):
	y_cols = ["a_sum_value", "a_max_value", "a_var_value", # 
			"b_max_value", "b_sum_value", "b_var_value"]
	for y_col in y_cols:
		one_map(df=df, y_col=y_col)

	# # normal join-distribution
	all_props = df.columns
	all_props = all_props.remove("bkg_volumes_original")
	prop_combs  = combinations(all_props, 2)
	for comb in prop_combs:
		xlabel = comb[0]
		ylabel = comb[1]
		X = df[[xlabel, ylabel]].values
		
		x = X[:, 0]
		y = X[:, 1]
		save_at = summary_folder + "/{0}/join_org/{1}.pdf".format(
							y_col, "|".join(comb))
		# joint_plot_2(x=x, y=y, 
		# 	xlabel=xlabel, ylabel=ylabel,
		# 	xlim=(min(x) - 0.1, max(x) + 0.1),
		# 	ylim=(min(y) - 0.1, max(y) + 0.1),
		# 	title=save_at.replace(ResultDir, ""), save_at=save_at)

		scatter_plot_4(x=x, y=y, color_array=None, xvlines=None, yhlines=None, 
				sigma=None, mode='scatter', lbl=None, name=None, 
				s=100, alphas=0.8, title=None,
				x_label='x', y_label='y', 
				save_file=save_at, 
				interpolate=False, color='blue', 
				preset_ax=None, linestyle='-.', marker='o')


if __name__ == "__main__":
	# pr_file = sys.argv[-1]
	pr_file = InputDir + "/params_grid/mcs10_msp100_a0.5_singleFalse.pkl"

	kwargs = load_pickle(filename=pr_file)
	print (kwargs)
	prefix = get_prefix(kwargs)

	particle_folder = ResultDir + prefix + text_dir + "/particles/" + job+  "/layer_0-546"
	summary_folder = ResultDir + prefix + "/task21"

	slice_3D_folder = ResultDir + prefix + "/slice_3D_bkg"
	reference_folder = InputDir + "/txt/Whole/fresh_CT13"

	summary_file = summary_folder + "/summary.csv"
	summary_df = pd.read_csv(summary_file, index_col=0)

	# # 1. create_params_grid
	map_particles(df=summary_df)