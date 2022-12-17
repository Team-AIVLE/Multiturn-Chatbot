import os
import gc
import json
import errno
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join as pjoin
from glob import iglob
from shutil import rmtree
from tqdm import tqdm

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def del_folder(path):
    try:
        rmtree(path)
    except:
        pass
    

def save_parquet(df, save_path):
    df.to_parquet(f'{save_path}.parquet')
    del df
    gc.collect()
    print(save_path, 'Done.')
     