#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:53:01 2017

@author: rishu
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 08:26:14 2016

@author: rishu
"""

import numpy as np
import bfast  as bf
import ewmacd as ew
import landTrendR  as ltr
from collections import defaultdict
from matplotlib import pylab as plt
import datetime as dt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

def data_from_timeSyncCsvFile(path, mat_all_lines, fn_groundTruth, my_pid, time_start, time_end):

    num_lines = len(mat_all_lines)
    this_pixel_info ={'sensor': [], 'pid':  [], 'yr': [],  \
                      'doy': [], 'b3': [], 'b4': [], 'b5': [], \
                      'b6': []}
    num_obs = 0
    for line in range(num_lines):
        mat_vals = mat_all_lines[line].strip().split(',')
        if (int(mat_vals[1]) == my_pid) and (mat_vals[0]!= '' and \
                      mat_vals[1]!= '' and mat_vals[2] != '' and \
                      mat_vals[4]!= '' and mat_vals[5]!='' and \
                      mat_vals[8]!='' and mat_vals[9]!='' and 
                      mat_vals[10]!='' and mat_vals[12]!=''): # and \
#                      int(mat_vals[13])==0):
            #sensor, pid, tsa, plotid, year, doy, band3, band4, band5, band6
                
            this_pixel_info['sensor'].append(int(mat_vals[0][2]))
            this_pixel_info['yr'].append(int(mat_vals[4]))
            this_pixel_info['doy'].append(int(mat_vals[5]))
            try:
                if (int(mat_vals[13]) not in [0, 1]):   # cloud, water etc masking
                    this_pixel_info['b3'].append(-9999)
                    this_pixel_info['b4'].append(-9999)
                    this_pixel_info['b5'].append(-9999)
                    this_pixel_info['b6'].append(-9999)
                else:
                    this_pixel_info['b3'].append(int(mat_vals[8]))
                    this_pixel_info['b4'].append(int(mat_vals[9]))
                    this_pixel_info['b5'].append(int(mat_vals[10]))
                    this_pixel_info['b6'].append(int(mat_vals[12]))
            except:
                    this_pixel_info['b3'].append(-9999)
                    this_pixel_info['b4'].append(-9999)
                    this_pixel_info['b5'].append(-9999)
                    this_pixel_info['b6'].append(-9999)
            num_obs +=1

    tyeardoy_all = np.zeros((num_obs, 2))
    vec_obs_all = []
    tyeardoy_all[:, 0] = this_pixel_info['yr']
    tyeardoy_all[:, 1] = this_pixel_info['doy']
    for i in range(num_obs):
        red = float(this_pixel_info['b3'][i])
        nir = float(this_pixel_info['b4'][i])
        if (abs(nir+red) < np.finfo(float).eps):
            vec_obs_all.append(-9999)
#            print 'i, ' , '-9999'
        else:
            ndvi = ((nir-red)/(nir+red))
            if ndvi < 0 or ndvi >1:
                vec_obs_all.append(-9999)
            else:
                vec_obs_all.append(ndvi * 10000)

    # limit returns to the desired time span
    a = [i for i in range(len(vec_obs_all)) \
            if (tyeardoy_all[i, 0] >= time_start) and (tyeardoy_all[i, 0] < time_end)]

    tyeardoy = np.zeros((len(a), 2))
    ctr = 0
    for i in a:
#        print tyeardoy_all[i, :]
        tyeardoy[ctr, 0] = int(tyeardoy_all[i, 0])
        tyeardoy[ctr, 1] = int(tyeardoy_all[i, 1])
        ctr += 1

    vec_obs = np.asarray([vec_obs_all[i] for i in a])
    # notice that tyeardoy is sent out as an array while vec_obs is sent as a list.
    # well, not any more
    
    # Now get the ground truths. Note that only disturbance pixels are included in the
    # ground truth sheet.
    mat_all_changes = []
    with open(fn_groundTruth, 'r') as f_gt:
        first_line = f_gt.readline()
        for i, line in enumerate(f_gt):
            this_line = line.strip().split(',')
            if int(this_line[0]) == my_pid:
                mat_all_changes.append(line)
            pass

    num_changes = 0
    changes = []
    # Note that if pid is a no-disturbance-ever pixel, then mat_all_changes will be empty anyways.
    # So this loop won't run. 
    # So, basically, the 'else' statement in the loop gets executed only for pids where some lu-cover info is available.
    for line in mat_all_changes:
        mat_gt_vals = line.strip().split(',')
        s_yr = mat_gt_vals[1]
        e_yr = mat_gt_vals[2]
        s_lu = mat_gt_vals[5]
        e_lu = mat_gt_vals[6]
        if ((int(s_yr) >= time_start) and (int(e_yr) <= time_end)):
            num_changes +=1
            change_type = mat_gt_vals[3]
            changes.append([s_yr, e_yr, change_type, s_lu, e_lu])
        else:
            changes.append(['x','x','x', s_lu, e_lu])
    changes = [num_changes] + changes

    return vec_obs, tyeardoy, changes

    
def process_pixel(tyeardoy, vec_obs_original, pixel_info, tickGap, changes, dist_thresh):
# this subroutine is for a fixed set of parameters. It is able to process multiple bands

    time1 = pixel_info[0]
    pid = pixel_info[1]
    num_bands = len(vec_obs_original)  # becuz vec_obs_original is a list
    num_obs = tyeardoy.shape[0]
    colors = ['green', 'sandybrown', 'black']
    dashes = [[12, 6, 12, 6], [12, 6, 3, 6], [3, 3, 3, 3]] # 10 points on, 5 off, 100 on, 5 off
#    line_styles = ['--','s--',':']
    line_widths = [4, 5, 2]

    # parameters for ewmacd
    ew_num_harmonics = 2  # was 2 earlier
    ew_ns = ew_num_harmonics
    ew_nc = ew_num_harmonics
    ew_xbarlimit1 = 1.5
    ew_xbarlimit2 = 20
    ew_lowthreshold = 0
    ew_trainingStart = time1
    ew_trainingEnd= time1 + 2
    ew_mu = 0
    ew_L = 3.0   # default is 3.0
    ew_lam = 0.5   # default is 0.5
    ew_persistence = 7  # default is 7
    ew_summaryMethod = 'on-the-fly'   #'reduced_wiggles'  # 'annual_mean'  #

    # parameters for bfast
    bf_h = 0.15
    bf_numBrks = 2
    bf_numColsProcess = 1
    bf_num_harmonics = 1
    bf_pval_thresh = 0.05  #default is 0.05
    bf_maxIter = 2
    bf_frequency = 23

    # parameters for landtrendr
    ltr_despike_tol = 0.9  # 1.0, 0.9, 0.75, default is 0.9
    ltr_pval = 0.2          # 0.05, 0.1, 0.2, default is 0.2
    ltr_mu = 6   #mpnu1     # 4, 5, 6, default is 6
    ltr_recovery_threshold = 1.0  # 1, 0.5, 0 25
    ltr_nu = 3              # 0, 3
    ltr_distwtfactor = 2.0   #i have taken this value from the IDL code
    ltr_use_fstat = 0        #0 means 'use p_of_f', '1' is for 'dont use p_of_f'.
                                   #So if use_fstat = 0, the code will use p_of_f.
    ltr_best_model_proportion = 0.75

    for band in range(num_bands):
        bf_brks_GI, bfast_brkpts, bfast_trendFit, bf_brkptsummary = \
                    bf.bfast(tyeardoy, vec_obs_original[band], \
                             ew_trainingStart, ew_trainingEnd, ew_lowthreshold, ew_num_harmonics, \
                             bf_frequency, bf_numBrks, bf_num_harmonics, bf_h, bf_numColsProcess, \
                             bf_pval_thresh, bf_maxIter)
                        
        if (bestModelInd == -9999):  # applicable to all three algos
            print bestModelInd
            return [], [], [], [], 'none'


    return bfast_brkpts[1:-1], ewma_brkpts[1:-1], ltr_brkpts[1:-1], polyAlgo_brkpts[1:-1], winner

def process_timesync_pixels(path = "/home/rishu/research/thesis/myCodes/thePolyalgorithm/"):
    
    fn_timeSync_pids = path + "timeSync_pids_change.csv"
    fn_timeSync_ts = path + 'conus_spectrals.csv'
    fn_timeSync_disturbance = path + 'ts_disturbance_segments.csv'

    pixels_list = []
    with open(fn_timeSync_pids, 'r') as f:
        for i, line in enumerate(f):
            line_vals = line.strip().split(',')
            pixel = int(line_vals[0])
            if pixel not in pixels_list:
                pixels_list.append(pixel)
            pass

    mat_all_lines = defaultdict(list)
    with open(fn_timeSync_ts, 'r') as f:
        first_line = f.readline()
        for i, line in enumerate(f):
            this_line = line.strip().split(',')
            mat_all_lines[int(this_line[1])].append(line)
            pass
            
    with open("gpu_pixels_yes.csv", "w") as gh:
        # do nothing. just opening a file
    fh.close()
    
    with open("gpu_pixels_no.csv", "w") as gh:
        # do nothing. just opening a file
    fh.close()

    num_change_pixels = 0
    num_stable_pixels = 0
    change_bf_says_change = 0
    change_bf_says_stable = 0
    num_stable_pixels = 0
    stable_bf_says_stable = 0
    stable_bf_says_change = 0
    zero_reading_pixels = []
    for pixel in pixels_list:  #40027038, 38029024]:  #pixels_list:
        time1 = 2000
        time2 = 2012 + 1   # stay same for all pixels. I've kept them here only to keep all parameters in one place
        tickGap = 2
        dist_thresh = 365
        my_pid = pixel  #int(pixel[0:-1])
        #######################################################################
        vec_obs_original_ndvi, tyeardoy, changes = data_from_timeSyncCsvFile( \
                            path, mat_all_lines[my_pid], fn_timeSync_disturbance, my_pid, time1, time2)

        if (len(tyeardoy) != len(vec_obs_original_ndvi)) or (len(tyeardoy) == 0):
            zero_reading_pixels.append(my_pid)
            continue

        vec_obs_original = [] 
        vec_obs_original.append(vec_obs_original_ndvi)
        # vec_obs_original is a list of arrays. Each array cors to 1 band.
        pixel_info = [time1, my_pid]        
        
        bfast_brkpts, ewma_brkpts, ltr_brkpts, polyAlgo_brkpts, winner = \
                process_pixel(tyeardoy, vec_obs_original, pixel_info, tickGap, changes, dist_thresh)


#        ######### determine the accuracy ########
        ground_truth_bps = []
#        changes = [s_yr, e_yr, change_type, s_lu, e_lu]
        for element in changes[1:]:
            try:
                ct = element[2]
                start_yr = int(element[0])
                end_yr = int(element[1])
                if ct != 'x':
#                if ct == 'Harvest':  # we are not going to do any case of Sit. 
                                  # So, even if an Sit follows a Harvest, it
                                  # will not be included in ground_truth_bps
                                  # This will prevent overcounting Harvest+Sit events.
                    ground_truth_bps.append([start_yr, end_yr])
                    num_true_brks += 1
            except:
                num_true_brks += 0 # basically do nothing

#       ########## counting change vs no change #########
        if changes[0] > 0:
            # timesync says there was a change
            num_change_pixels += 1
            # does bfast say so?
            if len(bfast_brkpts) > 0:
                # bf also says there was a change. So, success
                change_bf_says_change += 1
                with open('gpu_pixels_change.csv', 'a') as gh:
                    gh.write(str(my_pid) + '\n')
                gh.close()
            else:
                change_bf_says_stable += 1
                
                
        if changes[0] == 0:
            # timesync says pixel was stable
            # does bfast say so
            num_stable_pixels +=1
            if len(bfast_brkpts) == 0:
                # bf says pixel was stable
                stable_bf_says_stable += 1
                with open('gpu_pixels_stable.csv', 'a') as gh:
                    gh.write(str(my_pid) + '\n')
                gh.close()
            else:
                stable_bf_says_change += 1
                                
    
    return


