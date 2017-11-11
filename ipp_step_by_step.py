import parallel_test_impl as pti
import process_pixels_by_engine as ppe


num_pixels = 1000 #this is the total number of pixels
def process_pixels_by_engine(engine_id, num_engines):
    '''
    ipengine level handler for 
    processing pixels
    '''

    max_pixels_per_engine = int(num_pixels/num_engines)
    pixel_offset = max_pixels_per_engine * engine_id

    ppe.num_pixels_this_engine = min(max_pixels_per_engine, num_pixels - pixel_offset)
    
    ppe.process_pixels(pixel_offset)
    
