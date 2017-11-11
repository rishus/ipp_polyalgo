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

bfast_h = 0.05
bfast_numBrks = 2
bfast_numColsProcess = 1
bfast_num_harmonics = 1
bfast_pval_thresh = 0.7
bfast_maxIter = 1
bfast_frequency  = 23  #, bfast_fv = sampling_frequency(tyeardoy[:,0])

# parameters for landtrendr
landtrend_despike_tol = 0.5
landtrend_mu = 6   #mpnu1
landtrend_nu = 3
landtrend_distwtfactor = 2.0
landtrend_recovery_threshold = 1.0
landtrend_use_fstat = 0  #use_fstat = 'false'. So the code will use p_of_f.
landtrend_best_model_proportion = 0.5
landtrend_pval = 0.05   #, 0.1, 0.2


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


def process_pixel(pixel_data):

#    resids, ewma_summary, ewma_brkpts, ewmacd_pixel_output =    \
#                                                        ew.ewmacd(tyeardoy, pixel_data,   \
#                             ewma_num_harmonics, ewma_xbarlimit1, ewma_xbarlimit2, \
#                             ewma_lowthreshold, ewma_trainingStart, ewma_trainingEnd, \
#                             ewma_mu, ewmacd_L, ewma_lam, ewma_persistence, \
#                             ewma_summaryMethod, ewma_ns, ewma_nc, ewma_num_bands)

    bestModelInd, allmodels_LandTrend, ltr_brks_GI, ltr_brkpts, ltr_trendFit, ltr_brkptsummary = \
                        ltr.landTrend(tyeardoy, pixel_data, \
                          ewma_trainingStart, ewma_trainingEnd, ewma_lowthreshold, ewma_num_harmonics, \
                          landtrend_despike_tol, landtrend_mu, landtrend_nu, \
                          landtrend_distwtfactor, landtrend_recovery_threshold, \
                          landtrend_use_fstat, landtrend_best_model_proportion, \
                          landtrend_pval)


    return [ltr_brkpts[1:-1], ltr_brkptsummary]  #[ewma_brkpts[1:-1], ewmacd_pixel_output]
    
