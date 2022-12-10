import re
import random
import pandas as pd
import numpy as np
import warnings
from os.path import join as pjoin
warnings.filterwarnings(action='ignore')

'''
Description
-----------
전처리에 필요한 파라미터 지정
    args.use_valid: 검증 데이터 사용 여부, False일 때 validation data는 빈 값 저장
    args.test_ratio: 테스트 데이터 비율
    args.seed: random seed 고정 (19로 고정)
'''
def base_setting(args):
    args.use_valid = getattr(args, 'use_valid', True)
    args.test_ratio = getattr(args, 'test_ratio', 0.2)
    args.seed = getattr(args, 'seed', 19)

'''
Description
-----------
전처리 함수

    def del_newline(text : str)
        개행/탭 문자 공백 문자로 변경
    def del_special_char(text : str)
        느낌표, 물음표, 쉼표, 온점을 제외한 특수문자 삭제
    def repeat_normalize(text : str, num_repeats : int)
        반복 문자 개수 num_repeats으로 제한
    def del_duplicated_space(text : str)
        중복 공백 삭제
'''
def del_newline(text : str):
    return re.sub('[\s\n\t]+', ' ', text)

def del_special_char(text : str):
    return re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ,.?!~0-9a-zA-Z\s]+', '', text)

repeatchars_pattern = re.compile('(\D)\\1{2,}')
def repeat_normalize(text : str, num_repeats : int):
    if num_repeats > 0:
        text = repeatchars_pattern.sub('\\1' * num_repeats, text)
    return text

def del_duplicated_space(text : str):
    return re.sub('[\s]+', ' ', text)

def preprocess(text : str):
    proc_txt = del_newline(text)
    proc_txt = del_special_char(proc_txt)
    proc_txt = repeat_normalize(proc_txt, num_repeats=3)
    proc_txt = del_duplicated_space(proc_txt)
    return proc_txt.strip()

def processing(args, data):
    base_setting(args)

    # seed 고정
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f'Original Length of Data : {len(data)}')

    data['proc_query'] = list(map(preprocess, data['query']))
    data['proc_reply'] = list(map(preprocess, data['reply']))
    data.to_csv(pjoin(args.data_dir, 'proc_data.csv'), index=False)
    return data


'''
Description
-----------
전체 데이터를 train, valid, test로 분할하여 args.result_dir 내에 저장
'''
def split_dataset(args, data):

    data = data.sample(frac=1, random_state=args.seed)
    num_test = int(len(data) * args.test_ratio)

    test_idx = 2 * num_test if args.use_valid else num_test
    valid_idx = num_test if args.use_valid else 0

    valid = data.iloc[:valid_idx]
    test = data.iloc[valid_idx:test_idx]
    train = data.iloc[test_idx:]

    valid.to_csv(pjoin(args.result_dir, 'valid.csv'), index=False)
    test.to_csv(pjoin(args.result_dir, 'test.csv'), index=False)
    train.to_csv(pjoin(args.result_dir, 'train.csv'), index=False)

    print(f"Total Number of Data : {len(data)} -> {len(valid) + len(test) + len(train)}")
    return
