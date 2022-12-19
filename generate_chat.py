import torch
import argparse
import logging
import random
import numpy as np

import warnings
import transformers
import pytorch_lightning as pl

from utils.morp_utils import *
from auto_regressive_model import AutoRegressiveModel
from seq2seq_model import Seq2SeqModel
from eval import evaluation

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 19

warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def base_setting(args):
    args.k = getattr(args, 'k', 2)

'''
Description
-----------
시드 고정
'''
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Korean Multisession Dialogue Model')
    parser.add_argument('--input_folder',
                        type=str,
                        default='data')
                        
    parser.add_argument('--output_folder',
                        type=str,
                        default='result')

    parser.add_argument('--max_len',
                        type=int,
                        default=256)

    parser.add_argument('--reply_len',
                        type=int,
                        default=64)

    parser.add_argument('--model_type',
                        type=str,
                        default='gpt2',
                        choices=['gpt2', 'bart'])

    parser.add_argument('--model_pt',
                        type=str,
                        default=None)

    parser.add_argument("--gpuid", nargs='+', type=int, default=0)

    parser = AutoRegressiveModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    # random seed 고정
    set_seed(SEED)
    
    # max_len, k 설정
    base_setting(args)

    # testing finetuned language model
    with torch.cuda.device(args.gpuid[0]):
        evaluation(args)