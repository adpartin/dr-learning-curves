import argparse
import learningcurve as lc
from learningcurve.lrn_crv import LearningCurve
from utils.utils import load_data

parser = argparse.ArgumentParser(description='Generate LC vector.')
parser.add_argument('-dp', '--datapath', required=True, default=None, type=str,
                    help='Full path to data (default: None).')
parser.add_argument('--lc_step_scale', default='log', type=str, choices=['log', 'linear'],
                    help='Scale of progressive sampling of subset sizes in a learning curve (log2, log, log10, linear) (default: log).')
parser.add_argument('--min_size', default=128, type=int, help='The lower bound for the subset size (default: 128).')
parser.add_argument('--max_size', default=None, type=int, help='The upper bound for the subset size (default: None).')
parser.add_argument('--lc_sizes', default=5, type=int, help='Number of subset sizes (default: 5).')
parser.add_argument('--lc_sizes_arr', nargs='+', type=int, default=None, help='List of the actual sizes in the learning curve plot (default: None).')
args = parser.parse_args()
args = vars(args)

data = load_data(args['datapath'])

# Dummy data
ydata = data.iloc[:,0]
xdata = data.iloc[:,:2]

# LC args
lc_init_args = { 'lc_step_scale': args['lc_step_scale'],
                 'lc_sizes': args['lc_sizes'],
                 'min_size': args['min_size'],
                 'max_size': args['max_size'],
                 'lc_sizes_arr': args['lc_sizes_arr'],
                 'print_fn': print}
                
lc_obj = LearningCurve( X=xdata, Y=ydata, meta=None, **lc_init_args )
# print(lc_obj.tr_sizes)

