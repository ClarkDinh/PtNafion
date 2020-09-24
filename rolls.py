import numpy as np

def roll(x, k_step): 
	# work well with k_step = 1

	x_copy = x

	# go straight k_step
	x_up = roll_up(x, k_step)
	x_down = roll_down(x, k_step)
	x_left = roll_left(x, k_step)
	x_right = roll_right(x, k_step)


	x_up_left = roll_up(x_left, k_step)
	x_up_right = roll_up(x_right, k_step)
	x_down_left = roll_down(x_left, k_step)
	x_down_right = roll_down(x_right, k_step)

	outs = [x_up, x_down, x_left, x_right, x_up_left, x_up_right, x_down_left, x_down_right]
	out_rv = np.array([k.ravel() for k in outs])

	return out_rv



def roll_full_k(x, k_steps):
	# work for all with k_step = 1

	x_copy = x

	outs = [x]
	# go straight k_step in k_steps: total 4 * k_steps
	x_ups, x_downs, x_lefts, x_rights = [], [], [], []
	for k_step in range(k_steps):
		x_up = roll_up(x, k_step+1)
		x_down = roll_down(x, k_step+1)
		x_left = roll_left(x, k_step+1)
		x_right = roll_right(x, k_step+1)

		x_ups.append(x_up)
		x_downs.append(x_down)
		x_lefts.append(x_left)
		x_rights.append(x_right)

		outs.append(x_up)
		outs.append(x_down)
		outs.append(x_left)
		outs.append(x_right)

	# "up" i, "left" j; "up" i, "right" j; ... loop for all
	# total 4 * k_steps**2
	for i in range(k_steps):
		for j in range(k_steps):
			x_up_left = roll_up(x_lefts[i], j+1)
			x_up_right = roll_up(x_rights[i], j+1)
			x_down_left = roll_down(x_lefts[i], j+1)
			x_down_right = roll_down(x_rights[i], j+1)

			outs.append(x_up_left)
			outs.append(x_up_right)
			outs.append(x_down_left)
			outs.append(x_down_right)

	# in total :: 4*k_steps**2 + 4*k_steps
	# outs = [x_up, x_down, x_left, x_right, x_up_left, x_up_right, x_down_left, x_down_right]
	out_rv = np.array([k.ravel() for k in outs]).T

	return out_rv



def roll_outer_only_k(x, k_steps):
	# work for all with k_step = 1
	# get only outer region
	#
	# 1 1 1 1 1 1 
	# 1         1
	# 1         1
	# 1         1
	# 1 1 1 1 1 1
	x_copy = x
	outs = [x]
	# go straight k_step in k_steps: total 4 * k_steps
	x_ups, x_downs, x_lefts, x_rights = [], [], [], []
	for k_step in range(k_steps):
		x_up = roll_up(x, k_step+1)
		x_down = roll_down(x, k_step+1)
		x_left = roll_left(x, k_step+1)
		x_right = roll_right(x, k_step+1)

		x_ups.append(x_up)
		x_downs.append(x_down)
		x_lefts.append(x_left)
		x_rights.append(x_right)

		if k_step == k_steps - 1:
			outs.append(x_up)
			outs.append(x_down)
			outs.append(x_left)
			outs.append(x_right)

	# "up" i, "left" j; "up" i, "right" j; ... loop for all
	# total 4 * k_steps**2
	for i in range(k_steps):
		for j in range(k_steps):
			if i == k_steps -1 or j == k_steps -1:
				x_up_left = roll_up(x_lefts[i], j+1)
				x_up_right = roll_up(x_rights[i], j+1)
				x_down_left = roll_down(x_lefts[i], j+1)
				x_down_right = roll_down(x_rights[i], j+1)

				outs.append(x_up_left)
				outs.append(x_up_right)
				outs.append(x_down_left)
				outs.append(x_down_right)

	# in total :: 4*k_steps**2 + 4*k_steps
	# outs = [x_up, x_down, x_left, x_right, x_up_left, x_up_right, x_down_left, x_down_right]
	out_rv = np.array([k.ravel() for k in outs]).T

	return out_rv







def roll_up(x, k_step):
	x_up = np.roll(x, -k_step, axis=0)
	# x_up[-k_step:] = x[-k_step:] # duplicate value
	x_up[-k_step:] = np.nan


	return x_up

def roll_down(x, k_step):
	x_down = np.roll(x, k_step, axis=0)
	# x_down[:k_step] = x[:k_step]  # duplicate value
	# x_down[:k_step] = float('NaN')
	x_down[:k_step] = np.nan


	return x_down

def roll_left(x, k_step):
	x_left = np.roll(x, -k_step, axis=1)
	# x_left[:, -k_step:] = x[:, -k_step:]  # duplicate value
	x_left[:, -k_step:] = np.nan

	return x_left

def roll_right(x, k_step):
	x_right = np.roll(x, k_step, axis=1)
	# x_right[:, :k_step] = x[:, :k_step]  # duplicate value
	x_right[:, :k_step] = np.nan

	return x_right




if __name__ == "__main__":
	x = np.arange(15.0).reshape(3,5)
	print (x)



	env_list = []
	this_env = None
	for i in range(2):
		z_rols = roll_full_k(x, 1)
		# tmp = roll_full_k(x, 2)

		if this_env is None:
			this_env = z_rols
		else:
			this_env = np.concatenate((this_env, z_rols), axis=-1)

	# print (this_env)
	means = np.nanmean(this_env, axis=-1)
	print (means)


	stds = np.nanstd(this_env, axis=-1)
	print (stds)
	# for env in this_env:
	# 	print (np.isnan(env))
	# 	print (env, np.nanmean(env))	
	# 	print ("=====")
