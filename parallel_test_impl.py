import matplotlib as mpl
mpl.use('Agg')
from joblib import Parallel, delayed
import multiprocessing
import os
import numpy as np
import sys
#import pprofile
import math

from collections import defaultdict
import pickle

import call_ewmacd_v2 as call_ewmacd

num_rows = 50 #7411  #1000 #50
num_cols = 50 #8801  #1000 #50
num_pixels = num_rows * num_cols  #7411 * 8801   #1000*1000
input_file = '1836_Sandbox_50x50.timeOrder'    #'fullScene.timeOrder'   #    #'1836_Sandbox_1000x1000.timeOrder
out_file = input_file + "_ltr.out"
num_partitions = 1
num_time_points_per_pixel  = 198
total_timepoints  = None
max_pixels_per_partition = None
span_per_partition = None


num_cores = 1
def read_binary_part(ifn, ofn, offset, count):
    try:
        mdata = np.memmap(ifn, dtype='int16',  mode='r', offset=offset*2, shape=(count),) #because of int16 -- whyis offset in bytes
        #odata = np.memmap(ofn, dtype='int16',  mode='w+', offset=offset*2, shape=(count),) #because of int16 -- why is offset in bytes

        return mdata
    except Exception as e:
        print("Error in reading file..exiting", str(e))
        assert False
        sys.exit()


def processInput(partition_idx, res_arr):
    pixel_id_offset =  max_pixels_per_partition * partition_idx
    offset = span_per_partition * partition_idx
    time_points_this_partition = min(span_per_partition, total_timepoints - offset)
    idata = read_binary_part(input_file, out_file, offset, time_points_this_partition)
    all_brkpts = defaultdict(lambda  : defaultdict(list))
    assert time_points_this_partition%num_time_points_per_pixel == 0
    pixels_this_partition  = int(time_points_this_partition/num_time_points_per_pixel)

    for i in range(0, pixels_this_partition):
        pixel_data = idata[i * num_time_points_per_pixel: (i+1) * num_time_points_per_pixel]
        #pixel_out =  call_ewmacd.process_pixel(pixel_data)
        [brkpts, pixel_out] =  call_ewmacd.process_pixel(pixel_data)
        
        pixel_orig_id = pixel_id_offset + i
        col_id = int(pixel_orig_id/num_rows)
        row_id = int(pixel_orig_id%num_rows)
  	#print 'pixel_orig_id =', pixel_orig_id, 'px =', row_id, 'py =', col_id
        all_brkpts[row_id][col_id] = brkpts
        if i == 20:
            break
        res_arr[pixel_orig_id * num_time_points_per_pixel: (pixel_orig_id+1) * num_time_points_per_pixel] = pixel_data
    #pickle.dump( dict(all_brkpts), open("all_brkpts.p" + str(partition_idx), "wb" ) )
    return [pixels_this_partition , dict(all_brkpts)]

def initialize(dates_fn):
    global total_timepoints
    global  max_pixels_per_partition
    global span_per_partition
    
    total_timepoints  = num_time_points_per_pixel * num_pixels
    max_pixels_per_partition = int(math.ceil(float(num_pixels)/num_partitions))
    span_per_partition = max_pixels_per_partition * num_time_points_per_pixel
    call_ewmacd.dates_file_name = dates_fn
    call_ewmacd.initialize()

def process_pixels():
    odata = np.memmap(out_file, dtype='int16',  mode='w+', offset=0, shape=(num_pixels * num_time_points_per_pixel),)
    inputs = range(num_partitions)
    processInput(0, odata)
    par_res_out = Parallel(n_jobs=num_cores)(delayed(processInput)(i, odata) for i in inputs)
    
    # all_brkpts = defaultdict(lambda  : defaultdict(list))
    # for pixel_out, brkpt_dict in par_res_out:
    #     for  k in brkpt_dict.keys():
    #       all_brkpts[k].update(brkpt_dict[k])
    # all_brkpts = dict(all_brkpts)
    # pickle.dump( all_brkpts, open("all_brkpts_ltr.p", "wb" ) )


