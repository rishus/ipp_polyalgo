import pickle
par_res_out = None


num_change_pixels_g = 0
change_bf_says_change_g = 0
change_bf_says_stable_g = 0
change_ew_says_change_g = 0
change_ew_says_stable_g = 0
change_ltr_says_change_g = 0
change_ltr_says_stable_g = 0
change_poly_says_change_g = 0
change_poly_says_stable_g = 0
num_stable_pixels_g = 0
stable_bf_says_stable_g = 0
stable_bf_says_change_g = 0
stable_ew_says_stable_g = 0
stable_ew_says_change_g = 0
stable_ltr_says_stable_g = 0
stable_ltr_says_change_g = 0
stable_poly_says_stable_g = 0
stable_poly_says_change_g = 0
ew_g = 0
ltr_g = 0
bf_g = 0
nun_g = 0
insuff_g = 0

with open("par_res_out.bin2", "rb") as fh:
    par_res_out = pickle.load(fh)

for res_out in par_res_out:
    [num_change_pixels, num_stable_pixels, 
     change_bf_says_stable, change_bf_says_change, \
     stable_bf_says_change, stable_bf_says_stable, \
     change_ew_says_stable, change_ew_says_change, \
     stable_ew_says_change, stable_ew_says_stable, \
     change_ltr_says_stable, change_ltr_says_change, \
     stable_ltr_says_change, stable_ltr_says_stable, \
     change_poly_says_stable, change_poly_says_change, \
     stable_poly_says_change, stable_poly_says_stable, [ew, ltr, bf, nun, insuff], \
          time_elapsed ] = res_out

    num_change_pixels_g += num_change_pixels
    change_bf_says_stable_g += change_bf_says_stable
    change_bf_says_change_g += change_bf_says_change
    change_ew_says_stable_g += change_ew_says_stable
    change_ew_says_change_g += change_ew_says_change
    change_ltr_says_stable_g += change_ltr_says_stable
    change_ltr_says_change_g += change_ltr_says_change
    change_poly_says_stable_g += change_poly_says_stable
    change_poly_says_change_g += change_poly_says_change
    num_stable_pixels_g += num_stable_pixels
    stable_bf_says_stable_g += stable_bf_says_stable
    stable_bf_says_change_g += stable_bf_says_change
    stable_ew_says_stable_g += stable_ew_says_stable
    stable_ew_says_change_g += stable_ew_says_change
    stable_ltr_says_stable_g += stable_ltr_says_stable
    stable_ltr_says_change_g += stable_ltr_says_change
    stable_poly_says_stable_g += stable_poly_says_stable
    stable_poly_says_change_g += stable_poly_says_change
    ew_g += ew
    ltr_g += ltr    
    bf_g += bf
    nun_g += nun
    insuff_g += insuff

dist_thresh = float(sys.argv[1])
with open("polyalgo_distances_1988to2012_" + dist_thresh + ".csv", "a") as fh:
    fh.write('ew: ' + str(ew_g) + ', bf: ' + str(bf_g) + ', ltr:' + str(ltr_g) + \
                    ', none: ' + str(nun_g) + ', insuffsicient: ' + str(insuff_g) + '\n')
    fh.write('num_change_pixels =' + str(num_change_pixels_g) + '\n')
    fh.write('change_bf_says_change =' + str(change_bf_says_change_g) + '\n')
    fh.write('change_bf_says_stable =' + str(change_bf_says_stable_g) + '\n')
    fh.write('change_ew_says_change =' + str(change_ew_says_change_g) + '\n')
    fh.write('change_ew_says_stable =' + str(change_ew_says_stable_g) + '\n')
    fh.write('change_ltr_says_change =' + str(change_ltr_says_change_g) + '\n')
    fh.write('change_ltr_says_stable =' + str(change_ltr_says_stable_g) + '\n')
    fh.write('change_poly_says_change =' + str(change_poly_says_change_g) + '\n')
    fh.write('change_poly_says_stable =' + str(change_poly_says_stable_g) + '\n')
    fh.write('num_stable_pixels =' + str(num_stable_pixels_g) + '\n')
    fh.write('stable_bf_says_stable =' + str(stable_bf_says_stable_g) + '\n')
    fh.write('stable_bf_says_change =' + str(stable_bf_says_change_g) + '\n')
    fh.write('stable_ew_says_stable =' + str(stable_ew_says_stable_g) + '\n')
    fh.write('stable_ew_says_change =' + str(stable_ew_says_change_g) + '\n')
    fh.write('stable_ltr_says_stable =' + str(stable_ltr_says_stable_g) + '\n')
    fh.write('stable_ltr_says_change =' + str(stable_ltr_says_change_g) + '\n')
    fh.write('stable_poly_says_stable =' + str(stable_poly_says_stable_g) + '\n')
    fh.write('stable_poly_says_change =' + str(stable_poly_says_change_g) + '\n')

fh.close()

