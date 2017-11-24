#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 12:13:22 2017

@author: rishu
"""
import numpy as np
import ewmacd as ew
import bfast as bf
import landTrendR as ltr
from matplotlib.colors import LinearSegmentedColormap  # for custom colors
numTimeStamps = 198 
dates_file_name = ''
dates_info_str = []
ewma_num_harmonics = 2  # was 2 earlier
ewma_ns = ewma_num_harmonics
ewma_nc = ewma_num_harmonics
ewma_xbarlimit1 = 1.5
ewma_xbarlimit2 = 20
ewma_lowthreshold = 1
ewma_trainingStart = 2009
ewma_trainingEnd= 2012
ewma_mu = 0
ewma_L = 3   #0.5 for 432
ewma_lam = 0.5   #0.3 for 432
ewma_persistence = 7  # was 10 earlier
ewma_summaryMethod = 'on-the-fly'
ewma_num_bands = 1 #7

bf_h = 0.05
bf_numBrks = 2
bf_numColsProcess = 1
bf_num_harmonics = 1
bf_pval_thresh = 0.7
bf_maxIter = 1
bf_frequency  = 23  #, bf_fv = sampling_frequency(tyeardoy[:,0])

# parameters for ltrr
ltr_despike_tol = 0.5
ltr_mu = 6   #mpnu1
ltr_nu = 3
ltr_distwtfactor = 2.0
ltr_recovery_threshold = 1.0
ltr_use_fstat = 0  #use_fstat = 'false'. So the code will use p_of_f.
ltr_best_model_proportion = 0.5
ltr_pval = 0.05   #, 0.1, 0.2


ewma_brkptsummary = None
bfast_brkptsummary = None
ltr_brkptsummary = None
tyeardoy = None

def initialize():
    global ewma_brkptsummary 
    global bfast_brkptsummary 
    global ltr_brkptsummary
    global dates_info_str
    global tyeardoy
    with open(dates_file_name, 'r') as f:
        first_line = f.readline()
        for i, line in enumerate(f):
            dates_info_str.append(line)
            pass
        
    tyeardoy = np.zeros((len(dates_info_str), 2))
    for line in range(len(dates_info_str)):
        tyeardoy[line, 0] = int(dates_info_str[line][0:4])
        tyeardoy[line, 1] = int(dates_info_str[line][5:8])
    ewma_brkptsummary = -2222 * np.ones((numTimeStamps))
    bfast_brkptsummary = -2222 * np.ones(( numTimeStamps))
    ltr_brkptsummary = -2222 * np.ones((numTimeStamps))


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

    dAB = max(daB)
    
    return dAB


def process_pixel(pixel_data):

    resids, ewma_summary, ewma_brkpts, ewma_pixel_output =    \
                                                        ew.ewmacd(tyeardoy, pixel_data,   \
                             ewma_num_harmonics, ewma_xbarlimit1, ewma_xbarlimit2, \
                             ewma_lowthreshold, ewma_trainingStart, ewma_trainingEnd, \
                             ewma_mu, ewmacd_L, ewma_lam, ewma_persistence, \
                             ewma_summaryMethod, ewma_ns, ewma_nc, ewma_num_bands)

    bestModelInd, allmodels_LandTrend, ltr_brks_GI, ltr_brkpts, ltr_trendFit, ltr_brkptsummary = \
                        ltr.landTrend(tyeardoy, pixel_data, \
                          ewma_trainingStart, ewma_trainingEnd, ewma_lowthreshold, ewma_num_harmonics, \
                          ltr_despike_tol, ltr_mu, ltr_nu, \
                          ltr_distwtfactor, ltr_recovery_threshold, \
                          ltr_use_fstat, ltr_best_model_proportion, \
                          ltr_pval)

    bf_brks_GI, bf_brkpts, bf_trendFit, bfast_brkptsummary = \
                               bf.bfast(tyeardoy, vec_obs_original[band], \
                  ewma_trainingStart, ewma_trainingEnd, ewma_lowthreshold, ewma_num_harmonics, \
                  bf_frequency, bf_numBrks, bf_num_harmonics, bf_h, bf_numColsProcess, \
                  bf_pval_thresh, bf_maxIter)

   use_ewma = 'yes'
   for brk in bf_brkpts[1:]:
	if brk[0] in range(ewma_trainingStart, ewma_trainingEnd+1):
                use_ewma = 'no'
   
   dist_BL = dist(bf_brks_GI[1:-1], ltr_brks_GI[1:-1])
   dist_LB = dist(ltr_brks_GI[1:-1], bf_brks_GI[1:-1])
   if use_ewma == 'yes':
	dist_BE = dist(bf_brks_GI[1:-1], ew_brks_GI[1:-1])
	dist_EB = dist(ew_brks_GI[1:-1], bf_brks_GI[1:-1])
	dist_LE = dist(ltr_brks_GI[1:-1], ew_brks_GI[1:-1])
	dist_EL = dist(ew_brks_GI[1:-1], ltr_brks_GI[1:-1])
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
	   polyAlgo_brkpts = ewma_brkpts
	   winner = 'ew'
	else:
	   polyAlgo_brkpts = ltr_brkpts
	   winner = 'ltr'
    else:
	polyAlgo_brkpts = []
	winner = 'none'


    # returning only the breakpoints is probably enough. No need to return *_bkrptsummary.
    return [ewma_brkpts[1:-1], ewmacd_pixel_output]  #[ltr_brkpts[1:-1], ltr_brkptsummary]  #
#    return [ewma_brkpts[1:-1], ltr_brkpts[1:-1], bf_brkpts[1:-1], polyAlgo_brkpts[1:-1], winner]
 

   
