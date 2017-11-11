



def process_pixels(engine_id, pixel_offset, num_pixels, num_time_points_per_pixel=198):
    
    
    memmap_byte_offset = pixel_offset * num_time_points_per_pixel * 2 #memmap takes byte offset
    out_file = input_file + "_" + str(engine_id) + "_ltr.out"
    odata = np.memmap(out_file, dtype='int16',  mode='w+', offset=memmap_byte_offset, shape=(num_pixels * num_time_points_per_pixels),)
    
