# -*- coding: utf-8 -*-
'''Created by Tai Dinh
This file is used to sumarize the information of particles in bkg, a, b, c folders
'''
from ntpath import basename
import os, glob, ntpath, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import csv
from Rubber_constant import *
from multiprocessing import Pool

def list_folders(path):
    return sorted(glob.glob(os.path.join(path, '*')))

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)  # Removes all the subdirectories!
        os.makedirs(path)

def get_information(particle):
    t1 = time.time()
    bkg_volumes_original = 0
    bkg_volumes_after = 0
    bkg_surfaces = 0
    bkg_size_z = 0
    bkg_size_x = 0
    bkg_size_y = 0
    bkg_max_value = 0
    bkg_sum_value = 0
    bkg_var_value = 0
    a_volumes = 0
    a_max_value = 0
    a_sum_value = 0
    a_var_value = 0
    b_volumes = 0
    b_max_value = 0
    b_sum_value = 0
    b_var_value = 0
    c_volumes = 0
    c_max_value = 0
    c_sum_value = 0
    c_var_value = 0
    print(particle) 
    # First get the information for the particle in the bkg folder
    size =  basename(particle).split("_")[2].split("-")
    bkg_size_z = size[0]
    bkg_size_x = size[1]
    bkg_size_y = size[2]

    try:
        slice_1D_bkg = np.loadtxt("{0}/{1}.txt".format(slice_3D_folder,basename(particle)))
        
        bulk_positions = np.where(slice_1D_bkg==1)
        # print(len(bulk_positions[0]))
        bkg_volumes_after = len(bulk_positions[0])
        
        surface_positions = np.where(slice_1D_bkg==0)
        # print(len(surface_positions[0]))
        bkg_surfaces = len(surface_positions[0])

        #Then get other information for particles in a,b,c folders
        bkg_slices = sorted(glob.glob("{0}/*.txt".format(particle)))
        bkg_all_slices = []
        a_all_slices = []
        b_all_slices = []
        c_all_slices = []

        # count = 0
        # Traverse all slices of a particle.

        for ith, bkg_slice in enumerate(bkg_slices):
            print ("Processing slide: ", ith)
            bkg_array = np.loadtxt(bkg_slice)
            bkg_all_slices.append(bkg_array)
            bkg_bulk_positions = np.where(bkg_array>=threshold)
            bkg_volumes_in_slice = len(bkg_bulk_positions[0])
            bkg_volumes_original += bkg_volumes_in_slice
            bkg_sum_value += np.sum(bkg_array[bkg_array>=threshold])

            index = basename(bkg_slice).replace(".txt","").replace("_xrange","").replace("-_yrange","").split("_")
            slice_id = index[0]
            x_min = int(index[1].split("-")[0])-2
            x_max = int(index[1].split("-")[1])+2
            y_min = int(index[2].split("-")[0])-2
            y_max = int(index[2].split("-")[1])+2

            # For slices in "a" folder
            a_original_array = np.loadtxt("{0}/a0/{1}.txt".format(reference_folder,slice_id))
            assert a_original_array.shape == (1024, 1024)

            a_sub_array = a_original_array[y_min:y_max,x_min:x_max]
            assert a_sub_array.shape == bkg_array.shape
            a_copy = np.empty(a_sub_array.shape)
            a_copy[:] = np.nan
            a_copy[bkg_bulk_positions] = a_sub_array[bkg_bulk_positions]
            # print(a_sub_array)
            a_all_slices.append(a_copy)

            a_number_of_volumes = np.count_nonzero(~np.isnan(a_copy))
            a_volumes += a_number_of_volumes
            a_sum_value += np.sum(a_copy[~np.isnan(a_copy)])

            # For slices in "b" folder
            b_original_array = np.loadtxt("{0}/b0/{1}.txt".format(reference_folder,slice_id))
            assert b_original_array.shape == (1024, 1024)

            b_sub_array = b_original_array[y_min:y_max,x_min:x_max]
            assert b_sub_array.shape == bkg_array.shape
            b_copy = np.empty(b_sub_array.shape)
            b_copy[:] = np.nan
            b_copy[bkg_bulk_positions] = b_sub_array[bkg_bulk_positions]
            # print(b_sub_array)
            b_all_slices.append(b_copy)

            b_number_of_volumes = np.count_nonzero(~np.isnan(b_copy))
            b_volumes += b_number_of_volumes
            b_sum_value += np.sum(b_copy[~np.isnan(b_copy)])

            # For slices in "c" folder
            c_original_array = np.loadtxt("{0}/c0/{1}.txt".format(reference_folder,slice_id))
            assert c_original_array.shape == (1024, 1024)

            c_sub_array = c_original_array[y_min:y_max,x_min:x_max]
            assert c_sub_array.shape == bkg_array.shape
            c_copy = np.empty(c_sub_array.shape)
            c_copy[:] = np.nan
            c_copy[bkg_bulk_positions] = c_sub_array[bkg_bulk_positions]
            # print(c_sub_array)
            c_all_slices.append(c_copy)

            c_number_of_volumes = np.count_nonzero(~np.isnan(c_copy))
            c_volumes += c_number_of_volumes
            c_sum_value += np.sum(c_copy[~np.isnan(c_copy)])

            # # Test if read array correctly
            # np.savetxt("a_sub_arrray_{0}.txt".format(count),a_sub_array)
            # np.savetxt("b_sub_arrray_{0}.txt".format(count),b_sub_array)
            # np.savetxt("c_sub_arrray_{0}.txt".format(count),c_sub_array)
            # count += 1

        # Stack slices to find max and variance of each partice in [bkg, a, b, c] folders    
        bkg_slices_3D = np.stack(bkg_all_slices, axis = 0)
        print("bkg_3D shape:",bkg_slices_3D.shape)
        if np.isnan(bkg_slices_3D).all():
            bkg_max_value = "All Nan"
            bkg_var_value = "All Nan"
        else:
            bkg_max_value = np.nanmax(bkg_slices_3D)
            bkg_bulk_matrix = (bkg_slices_3D[bkg_slices_3D>=threshold]).ravel()
            bkg_var_value = np.nanvar(bkg_bulk_matrix)

        a_slices_3D = np.stack(a_all_slices, axis = 0)
        print("a_3D shape:",a_slices_3D.shape)
        if np.isnan(a_slices_3D).all():
            a_max_value = "All Nan"
            a_var_value = "All Nan"
        else:
            a_max_value = np.nanmax(a_slices_3D)
            a_var_value = np.nanvar(a_slices_3D.ravel())

        b_slices_3D = np.stack(b_all_slices, axis = 0)
        print("b_3D shape:",b_slices_3D.shape)
        if np.isnan(b_slices_3D).all():
            b_max_value = "All Nan"
            b_var_value = "All Nan"
        else:
            b_max_value = np.nanmax(b_slices_3D)
            b_var_value = np.nanvar(b_slices_3D.ravel())

        c_slices_3D = np.stack(c_all_slices, axis = 0)
        print("c_3D shape:",c_slices_3D.shape)
        if np.isnan(c_slices_3D).all():
            c_max_value = "All Nan"
            c_var_value = "All Nan"
        else:
            c_max_value = np.nanmax(c_slices_3D)
            c_var_value = np.nanvar(c_slices_3D.ravel())

        t2 = time.time()
        print("Take :", round(t2 - t1, 2), "seconds")

    except Exception as e:
        pass

    return bkg_volumes_original, bkg_volumes_after, bkg_surfaces, bkg_size_z, bkg_size_x, \
        bkg_size_y, bkg_max_value, bkg_sum_value, bkg_var_value, \
        a_volumes, a_max_value, a_sum_value, a_var_value, \
        b_volumes, b_max_value, b_sum_value, b_var_value, \
        c_volumes, c_max_value, c_sum_value, c_var_value



def summarize_information(input_folder, output_folder):
    summary_df = pd.DataFrame(columns=["particle_id", "bkg_volumes_original", "bkg_volumes_after", "bkg_surfaces", "bkg_size_z", "bkg_size_x", \
        "bkg_size_y", "bkg_max_value","bkg_sum_value","bkg_var_value", \
        "a_volumes", "a_max_value","a_sum_value","a_var_value", \
        "b_volumes", "b_max_value","b_sum_value","b_var_value", \
        "c_volumes", "c_max_value","c_sum_value","c_var_value"])
    particles_in_bkg = list_folders(input_folder)
    # Traverse all particles in bkg folder
        

    with Pool(16) as p:
        results = p.map(get_information, particles_in_bkg)
        

    for i in range(len(particles_in_bkg)):
        # For particles in "bkg" folder
        particle_in_bkg = particles_in_bkg[i]

        bkg_volumes_original, bkg_volumes_after, bkg_surfaces, bkg_size_z, bkg_size_x, \
            bkg_size_y, bkg_max_value, bkg_sum_value, bkg_var_value, \
            a_volumes, a_max_value, a_sum_value, a_var_value,\
            b_volumes, b_max_value, b_sum_value, b_var_value, \
            c_volumes, c_max_value, c_sum_value, c_var_value = results[i]


        summary_df.loc[i,"particle_id"] = basename(particle_in_bkg)
        summary_df.loc[i,"bkg_volumes_original"] = bkg_volumes_original
        summary_df.loc[i,"bkg_volumes_after"] = bkg_volumes_after
        summary_df.loc[i,"bkg_surfaces"] = bkg_surfaces
        summary_df.loc[i,"bkg_size_z"] = bkg_size_z
        summary_df.loc[i,"bkg_size_x"] = bkg_size_x
        summary_df.loc[i,"bkg_size_y"] = bkg_size_y
        summary_df.loc[i,"bkg_max_value"] = bkg_max_value
        summary_df.loc[i,"bkg_sum_value"] = bkg_sum_value
        summary_df.loc[i,"bkg_var_value"] = bkg_var_value
    
        # For particle in "a" folder
        summary_df.loc[i,"a_volumes"] = a_volumes
        summary_df.loc[i,"a_max_value"] = a_max_value
        summary_df.loc[i,"a_sum_value"] = a_sum_value
        summary_df.loc[i,"a_var_value"] = a_var_value

        # For particle in "b" folder
        summary_df.loc[i,"b_volumes"] = b_volumes
        summary_df.loc[i,"b_max_value"] = b_max_value
        summary_df.loc[i,"b_sum_value"] = b_sum_value
        summary_df.loc[i,"b_var_value"] = b_var_value

        # For particle in "c" folder
        summary_df.loc[i,"c_volumes"] = c_volumes
        summary_df.loc[i,"c_max_value"] = c_max_value
        summary_df.loc[i,"c_sum_value"] = c_sum_value
        summary_df.loc[i,"c_var_value"] = c_var_value

        summary_df.to_csv("{0}/summary.csv".format(output_folder), index=False)

if __name__ == "__main__":
    pr_file = sys.argv[-1]
    kwargs = load_pickle(filename=pr_file)
    print (kwargs)
    prefix = get_prefix(kwargs)
    input_folder = ResultDir + prefix + text_dir + "/particles/" + job+  "/layer_0-546"

    # # results from previous task
    slice_3D_folder = ResultDir + prefix + "/slice_3D_bkg"

    # # reference folder contains all "a", "b", "c" tasks
    reference_folder = InputDir + "/txt/Whole/fresh_CT13"

    output_folder = ResultDir + prefix + "/task21"


    makedirs(output_folder)
    
    summarize_information(input_folder, output_folder)
    print("Finish!!!")