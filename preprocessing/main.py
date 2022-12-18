import argparse
import random
import numpy as np

import warnings
from preprocess import preprocess_dataset

SEED = 19

warnings.filterwarnings(action='ignore')

'''
Description
-----------
시드 고정
'''
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
def base_setting(args):
    args.proc_folder = getattr(args, 'proc_folder', 'proc')
    args.origin_folder = getattr(args, 'origin_folder', 'origin')
    args.k = getattr(args, 'k', 2)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Korean Multisession Dialogue Model')
    parser.add_argument('--input_folder',
                        type=str,
                        default='data')
                        
    parser.add_argument('--k',
                        type=int,
                        default=2)

    parser.add_argument('--output_folder',
                        type=str,
                        default='result')

    args = parser.parse_args()

    # random seed 고정
    set_seed(SEED)
    base_setting(args)
    preprocess_dataset(args)
    
    