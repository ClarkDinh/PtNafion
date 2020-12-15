# -*- coding: utf-8 -*-
'''Created by Tai Dinh
This file is used to make label for bulk and background in a particle
'''
from ntpath import basename
import sys, os, glob, ntpath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import csv
from itertools import combinations
from scipy.spatial import distance
import plot
from Rubber_constant import *

def list_folders(path):
    return glob.glob(os.path.join(path, '*'))

def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path) 
	else:
		shutil.rmtree(path)  # Removes all the subdirectories!
		os.makedirs(path)

def check_surface(surface, threshold):
	surface_1D = surface.ravel()
	# print(surface_1D.shape)
	sum_volume = 0
	for volume in surface_1D:
		# print(volume)
		if volume >= threshold:
			sum_volume += 1
	return sum_volume

def label_bulk_bkg(input_folder, output_folder,threshold):
	folders = list_folders(input_folder)
	# Traverse all particles
	for particle in folders:
		print(basename(particle))
		makedirs("{0}/{1}".format(output_folder,basename(particle)))
		# print(particle)
		slices = sorted(glob.glob("{0}/*.txt".format(particle)))
		# Traverse all slices of a particle.
		for slice in slices:
			array = np.loadtxt(slice)
			array = np.where(array>=threshold,1,-1)
			# print(array)
			saveto = "{0}/{1}/{2}".format(output_folder,basename(particle),basename(slice))
			np.savetxt(saveto, array)

if __name__ == "__main__":
	pr_file = sys.argv[-1]
	kwargs = load_pickle(filename=pr_file)
	print (kwargs)
	prefix = get_prefix(kwargs)

	input_folder = ResultDir + prefix + text_dir + "/particles/" + job+  "/layer_0-546"
	output_folder = ResultDir + prefix + "/label"

	makedirs(output_folder)
	label_bulk_bkg(input_folder, output_folder,threshold)
		
	print("Finish!!!")





