import matplotlib as mpl
mpl.use('Agg')
from joblib import Parallel, delayed
import multiprocessing
import os
import numpy as np
import sys
#import pprofile
import math
import random

from collections import defaultdict
import pickle

import call_ewmacd_profile as call_ewmacd

def partition_in_chunks(end, num_chunks):
    k, m = divmod(end, num_chunks)
    return [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(num_chunks)]

num_rows = 1000 #7411  #1000 #50
num_cols = 1000 #8801  #1000 #50
num_pixels = num_rows * num_cols  #7411 * 8801   #1000*1000
input_file = '1836_Sandbox_1000x1000.timeOrder'    #'fullScene.timeOrder'   #    #'1836_Sandbox_1000x1000.timeOrder
out_file = input_file + "_ltr.out"
num_partitions = 1
num_time_points_per_pixel  = 198
total_timepoints  = None
max_pixels_per_partition = None
span_per_partition = None

num_groups = 1

num_cores = 1
def read_binary_part(ifn,  offset, count):
    try:
        mdata = np.memmap(ifn, dtype='int16',  mode='r', offset=offset*2, shape=(count),) #because of int16 -- whyis offset in bytes
        #odata = np.memmap(ofn, dtype='int16',  mode='w+', offset=offset*2, shape=(count),) #because of int16 -- why is offset in bytes

        return mdata
    except Exception as e:
        print("Error in reading file..exiting", str(e))
        assert False
        sys.exit()


def processInput(group_idx, pixel_chunks):
    '''
    pixel_chunk: the range this partition should process
    '''
    print("group_idx  = ", group_idx)
    print("chunks = ", pixel_chunks)
    all_brkpts_list = []
    ewma_brkpts_list = []
    ltr_brkpts_list = []
    bf_brkpts_list = []
    polyAlgo_brkpts_list = []
    for pixel_chunk in pixel_chunks:
        pixel_id_begin =  pixel_chunk[0]
        pixel_id_end  = pixel_chunk[1]
        offset = pixel_id_begin * num_time_points_per_pixel #offset in terms of timepoints
        time_points_this_partition = (pixel_id_end - pixel_id_begin) * num_time_points_per_pixel
        idata = read_binary_part(input_file,  offset, time_points_this_partition)
        for pixel_id in range(pixel_id_begin, pixel_id_end):
            local_pixel_id = pixel_id - pixel_id_begin
            pixel_data = idata[local_pixel_id * num_time_points_per_pixel: (local_pixel_id+1) * num_time_points_per_pixel]
            brkpts =  call_ewmacd.process_pixel(pixel_data)
        
            col_id = int(pixel_id/num_rows)
            row_id = int(pixel_id%num_rows)
            if len(brkpts[2]) == 0:
                print "for pixel =  ", pixel_id
                print "num brkpts ", len(brkpts[0]), " ", brkpts[2]

            if len(brkpts[0]) > 0: #for ewma
                ewma_brkpts_list.append((row_id, col_id, brkpts[0]))
            if len(brkpts[1]) > 0: #for ltr
                ltr_brkpts_list.append((row_id, col_id, brkpts[1]))
            if len(brkpts[2]) > 0: #for bf
                bf_brkpts_list.append((row_id, col_id, brkpts[2]))
            if len(brkpts[3]) > 0: #for polyAlgo
                polyAlgo_brkpts_list.append((row_id, col_id, brkpts[3]))
        
    return [ewma_brkpts_list, ltr_brkpts_list, bf_brkpts_list, polyAlgo_brkpts_list]

def initialize(dates_fn):
    global total_timepoints
    global  max_pixels_per_partition
    global span_per_partition
    
    total_timepoints  = num_time_points_per_pixel * num_pixels
    max_pixels_per_partition = int(math.ceil(float(num_pixels)/num_partitions))
    span_per_partition = max_pixels_per_partition * num_time_points_per_pixel
    call_ewmacd.dates_file_name = dates_fn
    call_ewmacd.initialize()

    return

def process_pixels():
    #odata = np.memmap(out_file, dtype='int16',  mode='w+', offset=0, shape=(num_pixels * num_time_points_per_pixel),)
    chunks = partition_in_chunks(num_pixels, num_partitions) #
    random.shuffle(chunks)
    num_groups = num_partitions
    chunk_groups = partition_in_chunks(num_partitions, num_groups)
    print chunks[0:16]
    print chunk_groups[0:16]
    inputs = range(num_groups)
    #print(chunks[-1])
    for partition_idx, chunk_group in list(zip(inputs,chunk_groups))[0:1]:
        processInput(partition_idx, chunks[chunk_group[0]:chunk_group[1]])
    
    #par_res_out = Parallel(n_jobs=num_cores)(delayed(processInput)(partition_idx, chunks[chunk_group[0]:chunk_group[1]]) for (partition_idx, chunk_group) in zip(inputs, chunk_groups))
    

    # all_brkpts = defaultdict(lambda  : defaultdict(list))
    # for brkpt_dict in par_res_out:
    #     for  k in brkpt_dict.keys():
    #       all_brkpts[k].update(brkpt_dict[k])
    # all_brkpts = dict(all_brkpts)
    
    #pickle.dump(par_res_out, open("all_brkpts_ltr.p", "wb" ) )
    return

