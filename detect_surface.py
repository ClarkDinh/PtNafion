# -*- coding: utf-8 -*-
'''Created by Tai Dinh
This file is used to get the surface in each particle
'''
from ntpath import basename, defpath
import sys, os, glob, ntpath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import csv
from itertools import combinations
from scipy.spatial import distance
from plot import plot_density
from Rubber_constant import *

def list_folders(path):
    return sorted(glob.glob(os.path.join(path, '*')))

def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path)
	else:
		shutil.rmtree(path)  # Removes all the subdirectories!
		os.makedirs(path)

def stack_slices(input_folder, output_folder, output_slice_3D):
	folders = list_folders(input_folder)
	# Traverse all particles
	for particle in folders:
		print(basename(particle))
		makedirs("{0}/{1}".format(output_folder,basename(particle)))
		all_slices = []
		# print(particle)
		slices = sorted(glob.glob("{0}/*.txt".format(particle)))
		# Traverse all slices of a particle.
		for slice in slices:
			array = np.loadtxt(slice)
			all_slices.append(array)
		slices_3D = np.stack(all_slices, axis = 0)
		# print(slices_3D)
		# print("Slice_3D shape:",slices_3D.shape)
		positions = np.where(slices_3D[:,:,:]==1)
		positions = np.array(positions)
		#print(positions)
		# print("Bulk positions shape:",positions.shape)
		depth,rows,cols = slices_3D.shape
		# print(depth,rows, cols)
		for i in range(0,positions.shape[1]):
			d = positions[0][i]
			n = positions[1][i]
			m = positions[2][i]
			d_front = 0
			d_back = 0
			n_above = 0
			n_bottom = 0
			m_left = 0
			m_right = 0

			#For depth
			if d>0:
				d_front = d-1
				d_back = d+1
			if d == 0: #If the pixel lays on the first slice of the particle
				d_front = d
				d_back = d+1
			if d == depth-1: #If the pixel lays on the last slice of the particle
				d_front = d-1
				d_back = d

			#For height
			if n>0:
				n_above = n-1
				n_bottom = n+1
			if n == 0: #If the pixel lays on the top margin of the slice
				n_above = n
				n_bottom = n+1
			if n == rows-1: #If the pixel lays on the bottom margin of the slice
				n_above = n-1
				n_bottom = n
			# For width
			if m>0:
				m_left = m-1
				m_right = m+1
			if m == 0: #If the pixel lays on the left margin of the slice
				m_left = m
				m_right = m+1
			if m == cols-1: #If the pixel lays on the right margin of the slice
				m_left = m-1
				m_right  = m
			
			cubic = slices_3D[d_front:d_back+1, n_above:n_bottom+1, m_left:m_right+1]
			# print(d_front,d_back,n_above,n_bottom,m_left,m_right)
			# print(cubic)
			# print(cubic.shape)
			if -1 in cubic:
				slices_3D[d, n, m] = 0
		# Write slices_3D into txt file
		save_txt_3D = "{0}/{1}.txt".format(output_slice_3D,basename(particle))
		np.savetxt(save_txt_3D, slices_3D.ravel())

		if is_plot:
			for j in range(0,depth):
				slice_2D = slices_3D[j,:,:]
				# save_txt_2D = "{0}/{1}/{2}".format(output_folder,basename(particle),basename(slices[j]))
				# np.savetxt(save_txt_2D, slice_2D)
				img_name = basename(slices[j])
				img_name = img_name.replace(".txt","")
				save_img = "{0}/{1}/{2}".format(output_folder,basename(particle),img_name)
				# plot_density(slice_2D, save_img + ".pdf", cmap_name = "bwr", title=img_name, vmin=-1, vmax=1, is_save2input=None)
				plot_density(slice_2D, save_img + ".jpg", cmap_name = "bwr", 
					title=img_name, vmin=-1, vmax=1, is_save2input=None)

if __name__ == "__main__":
	pr_file = sys.argv[-1]
	kwargs = load_pickle(filename=pr_file)
	print (kwargs)
	prefix = get_prefix(kwargs)

	input_folder = ResultDir + prefix + "/label"
	output_folder = ResultDir + prefix +"/task22"
	output_slice_3D = ResultDir + prefix + "/slice_3D_bkg"

	makedirs(output_folder)
	makedirs(output_slice_3D)
	stack_slices(input_folder, output_folder, output_slice_3D)
		
	print("Finish!!!")







