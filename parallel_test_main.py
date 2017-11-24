import  argparse
import parallel_test_impl as pti
import time
import timeit

parser = argparse.ArgumentParser()
parser.add_argument("--num_pixels", help="total number of pixels", nargs=1)
parser.add_argument("--input_file", help="the data file ", nargs=1)
parser.add_argument("--num_partition", help="chunks", nargs=1)
parser.add_argument("--num_groups", help="num groups", nargs=1)

parser.add_argument("--dates_file", help="dates file name", nargs=1)
parser.add_argument("--num_cores", help="parallel workers", nargs=1)

args = parser.parse_args()


if args.num_pixels:
    pti.num_pixels = int(args.num_pixels[0])


if args.input_file:
    pti.input_file = args.input_file[0]


if args.num_partition:
    pti.num_partitions = int(args.num_partition[0])

if args.num_groups:
    pti.num_groups = int(args.num_groups[0])

dates_file = 'Dates_1837_198b.csv'
if args.dates_file:
    dates_file = args.dates_file[0]


if args.num_cores:
    pti.num_cores = int(args.num_cores[0])


start_time = timeit.default_timer()
pti.initialize(dates_file)
pti.process_pixels()
elapsed = timeit.default_timer() - start_time
print(elapsed)

