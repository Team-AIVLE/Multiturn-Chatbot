
import os
import errno

import pandas as pd
import numpy as np

from shutil import rmtree
from multiprocessing import Pool
from functools import partial

'''
Description
-----------
멀티 프로세싱을 수행하기 위한 함수로, 
data를 num_cores 개로 분할하여 각 프로세스 별로 func 수행
'''
def parallelize_dataframe(data, func, num_cores, args):
    sub_data = np.array_split(data, num_cores)
    pool = Pool(num_cores)

    data = pd.concat(pool.map(partial(func, args), sub_data))
    pool.close()
    pool.join()
    return data

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