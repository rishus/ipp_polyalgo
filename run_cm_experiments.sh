#!/bin/bash

for distThresh in `echo 1.0 1.5 2.5`
    do 
	python2  poly_for_timesync_v3_par.py $distThresh >  poly_for_timesync_v3_par_"$distThresh".out
        python2  postprocess_poly_for_timesync_v3_par.py $distThresh
    done

