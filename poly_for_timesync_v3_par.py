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
import ewmacd as ew
import landTrendR  as ltr
from joblib import Parallel, delayed
import multiprocessing
from collections import defaultdict
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
import bfast as bf
import datetime as dt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import pickle
import time
import sys

time1 = 1988
time2 = 2012 + 1
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

dist_thresh = 2.0

def partition_in_chunks(end, num_chunks):
    k, m = divmod(end, num_chunks)
    return [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(num_chunks)]


def data_from_timeSyncCsvFile(path, mat_all_lines, fn_groundTruth, pixel_id, time_start, time_end):
    '''
    this mat_all_lines is different from mat_all_lines
    '''

    num_lines = len(mat_all_lines)
    this_pixel_info ={'sensor': [], 'pid':  [], 'yr': [],  \
                      'doy': [], 'b3': [], 'b4': [], 'b5': [], \
                      'b6': []}
    num_obs = 0
    
    for line in range(num_lines):
        mat_vals = mat_all_lines[line].strip().split(',')
        if (int(mat_vals[1]) == pixel_id) and (mat_vals[0]!= '' and \
                                               mat_vals[1]!= '' and mat_vals[2] != '' and \
                                               mat_vals[4]!= '' and mat_vals[5]!='' and \
                                               mat_vals[8]!='' and mat_vals[9]!='' and 
                                               mat_vals[10]!='' and mat_vals[12]!=''): 
                
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
            if int(this_line[0]) == pixel_id:
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

    #time1 = pixel_info[0]
    pid = pixel_info[1]
    num_bands = len(vec_obs_original)  # becuz vec_obs_original is a list
    num_obs = tyeardoy.shape[0]
    colors = ['green', 'sandybrown', 'black']
    dashes = [[12, 6, 12, 6], [12, 6, 3, 6], [3, 3, 3, 3]] # 10 points on, 5 off, 100 on, 5 off
#    line_styles = ['--','s--',':']
    line_widths = [4, 5, 2]

    for band in range(num_bands):
        #bf_brks_GI, bfast_brkpts, bfast_trendFit, bf_brkptsummary = \
        bf_brks_GI, bf_brkpts, bf_trendFit = \
                    bf.bfast(tyeardoy, vec_obs_original[band], \
                             ew_trainingStart, ew_trainingEnd, ew_lowthreshold, ew_num_harmonics, \
                             bf_frequency, bf_numBrks, bf_num_harmonics, bf_h, bf_numColsProcess, \
                             bf_pval_thresh, bf_maxIter)
    	#tmp2, ewma_summary, ew_brks_GI, ew_brkpts, ew_brkptsummary =    \
    	tmp2, ewma_summary, ew_brks_GI, ew_brkpts = \
                                 ew.ewmacd(tyeardoy, vec_obs_original[band],  \
                                 ew_num_harmonics, ew_xbarlimit1, ew_xbarlimit2, \
                                 ew_lowthreshold, ew_trainingStart, ew_trainingEnd, \
                                 ew_mu, ew_L, ew_lam, ew_persistence, \
                                 ew_summaryMethod, ew_ns, ew_nc, 'dummy')

	#bestModelInd, allmodels_LandTrend, ltr_brks_GI, ltr_brkpts, ltr_trendFit, ltr_brkptsummary = \
	bestModelInd, allmodels_LandTrend, ltr_brks_GI, ltr_brkpts, ltr_trendFit = \
                        ltr.landTrend(tyeardoy, vec_obs_original[band], \
                          ew_trainingStart, ew_trainingEnd, ew_lowthreshold, ew_num_harmonics, \
                          ltr_despike_tol, ltr_mu, ltr_nu, ltr_distwtfactor, \
                          ltr_recovery_threshold, ltr_use_fstat, ltr_best_model_proportion, \
                          ltr_pval)

	if (bestModelInd == -9999):  # applicable to all three algos
        # happens when len training period is not sufficient
#            print bestModelInd
            return [], [], [], [], 'insuff'

	        ####### compare different algorithm results #########
        # pairwise distance between brkpts
        # Does any algorithm indicate breakpoint during the training period?
        use_ewma = 'yes'
        for brk in bf_brkpts[1:]:
            if brk[0] in range(ew_trainingStart, ew_trainingEnd+1):
                use_ewma = 'no'

	bf_brkpts_m = [(i[0] + min(i[1], 365)/365.) for i in bf_brkpts[1:-1]]
	ltr_brkpts_m = [(i[0] + min(i[1], 365)/365.) for i in ltr_brkpts[1:-1]]
	ew_brkpts_m = [(i[0] + min(i[1], 365)/365.) for i in ew_brkpts[1:-1]]
	dist_BL = dist(bf_brkpts_m, ltr_brkpts_m, 'no')
        dist_LB = dist(ltr_brkpts_m, bf_brkpts_m, 'no')
	if use_ewma == 'yes':
            dist_BE = dist(bf_brkpts_m, ew_brkpts_m,'no')
            dist_EB = dist(ew_brkpts_m, bf_brkpts_m,'no')
            dist_LE = dist(ltr_brkpts_m, ew_brkpts_m, 'no')
            dist_EL = dist(ew_brkpts_m, ltr_brkpts_m, 'no')
        else:
            dist_BE = 1000000
            dist_EB = 1000000        
            dist_LE = 1000000
            dist_EL = 1000000

	vec_dists = [dist_BE, dist_EB, dist_BL, dist_LB, dist_LE, dist_EL]
	s_ind = vec_dists.index(min(vec_dists))
	if vec_dists[s_ind] <= dist_thresh:
            if s_ind in [0, 2]:
                polyAlgo_brkpts = bf_brkpts
                winner = 'bf'
            elif s_ind in [1, 5]:
                polyAlgo_brkpts = ew_brkpts
                winner = 'ew'
            else:  # s_ind in [3, 4]
                polyAlgo_brkpts = ltr_brkpts
                winner = 'ltr'
        else:
            # if no agreement, then declare it stable
            polyAlgo_brkpts = [0, num_obs-1]
            winner = 'none'


    return bf_brkpts[1:-1], ew_brkpts[1:-1], ltr_brkpts[1:-1], polyAlgo_brkpts[1:-1], winner


def dist(A, B, toprint):

    lA = len(A)
    lB = len(B)
    if lA == 0 and lB == 0:
        return 0
        
    if lA == 0 and lB != 0:
        return 1000000
    if lB == 0 and lA != 0:
        return 1000000

    dists_AB = np.zeros((lA, lB))
    daB = np.zeros( (lA,) )
    for i in range(lA):
        # get d(a, B)
        for j in range(lB):
            dists_AB[i, j] = abs(A[i]-B[j])
            
        daB[i] = min(dists_AB[i,:])
    if toprint == 'bl':
        print 'BFAST, LTR:'
    if toprint == 'lb':
        print 'LTR, BFAST'
    if toprint in ['bl','lb']:
        print 'A: ', A
        print 'B: ', B
        for i in range(lA):
            print dists_AB[i,:]
        
        print 'daB = ', daB

    dAB = max(daB)
    if toprint in ['bl','lb']:
        print 'dAB = ', dAB
    
    return dAB


def load_conus_spectral(data_dir, pixels_list):
    mat_all_lines = defaultdict(list)
    fn_timeSync_ts = data_dir + '/conus_spectrals.csv'
    with open(fn_timeSync_ts, 'r') as f:
        first_line = f.readline()
        for i, line in enumerate(f):
            this_line = line.strip().split(',')
	    if int(this_line[1]) in pixels_list:
	            mat_all_lines[int(this_line[1])].append(line)
            pass
    return mat_all_lines


def load_timesync_pids(data_dir):
    pixels_list = []    
    fn_timeSync_pids = data_dir + "/ts_disturbance_segments_allpids.csv"
    #fn_timeSync_pids = data_dir + "/timeSync_pids.csv"
    with open(fn_timeSync_pids, 'r') as f:
        for i, line in enumerate(f):
            line_vals = line.strip().split(',')
            pixel = int(line_vals[0])
            if pixel not in pixels_list:
                pixels_list.append(pixel)
            pass
    return pixels_list


def parprocess(pixel_list, data_dir, fn_timeSync_disturbance, time1, time2, tickGap, dist_thresh):

    num_change_pixels = 0
    change_bf_says_change = 0
    change_bf_says_stable = 0
    change_ew_says_change = 0
    change_ew_says_stable = 0
    change_ltr_says_change = 0
    change_ltr_says_stable = 0
    change_poly_says_change = 0
    change_poly_says_stable = 0
    num_stable_pixels = 0
    stable_bf_says_stable = 0
    stable_bf_says_change = 0
    stable_ew_says_stable = 0
    stable_ew_says_change = 0
    stable_ltr_says_stable = 0
    stable_ltr_says_change = 0
    stable_poly_says_stable = 0
    stable_poly_says_change = 0
    #print "starting reading mat_all_lines"
    mat_all_lines =  load_conus_spectral(data_dir, pixel_list)
    #print "done loading mat_all_lines"
    ew = 0
    ltr = 0
    bf = 0
    nun = 0
    insuff = 0
    problem_pixels = []
    start = time.time()
    for pixel_id in pixel_list:  #40027038, 38029024]:  #pixels_list:

        vec_obs_original_ndvi, tyeardoy, changes = data_from_timeSyncCsvFile( \
                   data_dir, mat_all_lines[pixel_id], fn_timeSync_disturbance, \
                   pixel_id, time1, time2)

        if (len(tyeardoy) != len(vec_obs_original_ndvi)) or (len(tyeardoy) == 0):
		problem_pixels.append(pixel_id)
		continue

        vec_obs_original = [] 
        vec_obs_original.append(vec_obs_original_ndvi)
        pixel_info = [time1, pixel_id]        
        #print "processing pixel = ", pixel_id
	try:
            bfast_brkpts, ewma_brkpts, ltr_brkpts, polyAlgo_brkpts, winner = \
                                                                             process_pixel(tyeardoy, vec_obs_original, pixel_info, \
                                                                                           tickGap, changes, dist_thresh)
        except:
            problem_pixels.append(pixel_id)	
            continue

#       ########## counting selections #########
        if winner == 'ew':
            ew += 1
        if winner == 'bf':
            bf += 1
        if winner == 'ltr':
            ltr += 1
        if winner == 'none':
            nun += 1
        if winner == 'insuff':
            insuff += 1

#       ########## counting change vs no change #########
        if changes[0] > 0:
            # timesync says there was a change
            num_change_pixels += 1
            # does bfast say so?
            if len(bfast_brkpts) > 0:
                # bf also says there was a change. So, success
                change_bf_says_change += 1
            else:
                change_bf_says_stable += 1
	    
            if len(ewma_brkpts) > 0:
                change_ew_says_change += 1
            else:
                change_ew_says_stable += 1
                
            if len(ltr_brkpts) > 0:
                change_ltr_says_change += 1
            else:
                change_ltr_says_stable += 1
                
            if len(polyAlgo_brkpts) > 0:
                change_poly_says_change += 1
            else:
                change_poly_says_stable += 1
                
        if changes[0] == 0:
            # timesync says pixel was stable
            # does bfast say so
            num_stable_pixels +=1
            if len(bfast_brkpts) == 0:
                # bf says pixel was stable
                stable_bf_says_stable += 1
            else:
                stable_bf_says_change += 1

            if len(ewma_brkpts) == 0:
                stable_ew_says_stable += 1
            else:
                stable_ew_says_change += 1
                
            if len(ltr_brkpts) == 0:
                stable_ltr_says_stable += 1
            else:
                stable_ltr_says_change += 1
            
            if len(polyAlgo_brkpts) == 0:
                stable_poly_says_stable += 1
            else:
                stable_poly_says_change += 1
#        print "pixel_id = ", pixel_id
#        print "changes = ", changes
#        print num_change_pixels, num_stable_pixels
    end = time.time()
    time_elapsed = end - start
    print 'done with', len(pixel_list),  'process; time_take= ', time_elapsed

    return [num_change_pixels, num_stable_pixels, 
          		       change_bf_says_stable, change_bf_says_change, \
                               stable_bf_says_change, stable_bf_says_stable, \
                               change_ew_says_stable, change_ew_says_change, \
                               stable_ew_says_change, stable_ew_says_stable, \
                               change_ltr_says_stable, change_ltr_says_change, \
                               stable_ltr_says_change, stable_ltr_says_stable, \
                               change_poly_says_stable, change_poly_says_change, \
                               stable_poly_says_change, stable_poly_says_stable, \
                   	       [ew, ltr, bf, nun, insuff], time_elapsed ]


def process_timesync_pixels(data_dir = "/home/t/ltw/big_files_rishu/optimizing_parallization", pixel_list = None):

    global dist_thresh
    # if pixel_list is None:
    #     pixel_list = load_timesync_pids(data_dir)
    
    # if mat_all_lines is None:
    #     mat_all_lines = load_conus_spectral(data_dir)
    #gen list of all the pixels
    

    #assemble details of each pid in mat_all_lines

    fn_timeSync_disturbance = data_dir + '/ts_disturbance_segments.csv'


    tickGap = 2

    gpc  = open('gpu_pixels_change.csv', 'w')
    gps = open('gpu_pixels_stable.csv', 'w')

    num_pixels = len(pixel_list)
    num_partitions = 200
    num_cores = 55
    chunks = partition_in_chunks(num_pixels, num_partitions)
    par_res_out = Parallel(n_jobs=num_cores)(delayed(parprocess)(pixel_list[chunk[0]:chunk[1]], data_dir,  fn_timeSync_disturbance,  time1, time2, tickGap, dist_thresh) for chunk in chunks)
    
    print 'going to write the pickle file'
    with open("par_res_out_" + str(dist_thresh)+".bin2", "wb") as fh:
        pickle.dump( par_res_out, fh)
    #return par_res_out

data_dir = "/home/t/ltw/big_files_rishu/optimizing_parallization"


dist_thresh = float(sys.argv[1])
pixel_list = load_timesync_pids(data_dir)
print "num pixels = ", len(pixel_list)
start = time.time()
process_timesync_pixels(pixel_list = pixel_list)
end = time.time()
print("time taken", end - start)

#def main():
#	process_timesync_pixels(dist_thresh)

#if __name__ == '__main__':
#	main


