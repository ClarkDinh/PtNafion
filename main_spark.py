from Rubber_constant import *
from itertools import product


def main():
	is_partition = False
	is_summary_info = True

	min_cluster_size_list = [10, 50, 100, 200, 500]
	min_samples_list = [10, 50, 100, 200, 500]
	allow_single_cluster_list = [True, False] # 
	alpha_list = [ 0.2, 0.5, 0.8 ] # 

	all_kwargs = list(product(min_cluster_size_list, min_samples_list, allow_single_cluster_list, alpha_list))
	n_tasks = len(all_kwargs)

	cpus_per_task = 16
	max_cpus = 9*32 # # ncpus take * ncores per cpu
	ntask_per_batch = int(max_cpus / cpus_per_task)

	nbatch = int(n_tasks/ntask_per_batch)
	makedirs(MainDir+"/input/batch_list/tmps.txt")

	for batch_ith in range(nbatch):
		shrun_file = open(MainDir+"/input/batch_list/batch_run_{0}.sh".format(batch_ith),"w") 
		shrun_file.write("#!/bin/bash \n")
		shrun_file.write("#SBATCH --ntasks={0}\n".format(ntask_per_batch))
		shrun_file.write("#SBATCH --output=./output_{0}.txt\n".format(batch_ith))
		
		shrun_file.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
		# shrun_file.write("#SBATCH --mem-per-cpu=16000\n")

		init_kw = batch_ith*ntask_per_batch
		last_kw = (batch_ith+1)*ntask_per_batch

		for kw in all_kwargs[init_kw:last_kw]:
			min_cluster_size, min_samples, allow, alpha = kw[0], kw[1], kw[2], kw[3]
			kwargs = dict(
				{"min_cluster_size":min_cluster_size, 
				"min_samples":min_samples, 
				"alpha":alpha, "allow_single_cluster":allow})

			param_file = InputDir +"/params_grid/mcs{0}_msp{1}_a{2}_single{3}.pkl".format(
								min_cluster_size, min_samples, alpha, allow)
			makedirs(param_file)
			dump_pickle(data=kwargs, filename=param_file)
			sh_file = MainDir+"/input/sh/mcs{0}_msp{1}_a{2}_single{3}.sh".format(
								min_cluster_size, min_samples, alpha, allow)
			makedirs(sh_file)

			with open(sh_file, "w") as f:
				f.write("cd {0}\n".format(MainDir+"/code"))
				if is_partition:
					f.write("python partitioning.py {0}\n".format(param_file))
				
				if is_summary_info:
					# # 0.0
					f.write("python label_data.py {0}\n".format(param_file))
					# # 1.0
					f.write("python detect_surface.py {0}\n".format(param_file))
					# # 2.0
					f.write("python summary_information.py {0}\n".format(param_file))

			shrun_file.write("srun --ntasks=1 --nodes=1 sh {0} &\n".format(sh_file))
		shrun_file.write("wait\n")
		shrun_file.close()

# conda install --force-reinstall -y -q --name ctxafs -c conda-forge --file requirements.txt


if __name__ == "__main__":

	# # 1. create_params_grid
	main()




