import json
import pandas as pd

from os.path import join as pjoin
from glob import iglob

from tqdm import tqdm
from data_utils import *
from morp_utils import *

P = {
    'valid' : 'validation/format_train',
    'train' : 'train'
}
DELIMITER = '<unused1>'

def json_to_parquet(args):
    train_persona = {}
    valid_persona = {}

    mkdir_p(args.origin_folder)

    for dir_p in list(iglob(pjoin(args.input_folder, "session_**/"))):
        sess_lv = dir_p.split("/")[-2]

        for d_type in ['train', 'valid']:
            dtype_dir_p = pjoin(args.origin_folder, d_type)
            mkdir_p(dtype_dir_p)
            
            df = pd.DataFrame()
            chat_paths = list(iglob(pjoin(dir_p, f"{P[d_type]}/**.json")))
            for chat_p in tqdm(chat_paths, desc=f"{sess_lv}", total=len(chat_paths)):
                
                with open(chat_p, encoding="UTF-8") as f:
                    entire_info = json.load(f)
                    
                    # speaker1의 페르소나 추출
                    persona = entire_info['personaInfo']['clInfo']['personaFeatures']
                    persona_kw = ' '.join(extract_keyword(persona))
                    
                    filename = entire_info["FileInfo"]["filename"]
                    
                    for sess in entire_info['sessionInfo']:
                        # Get prev-session information                      
                        if d_type == 'train':
                            train_persona[sess['sessionID']] = persona_kw
                        else:
                            valid_persona[sess['sessionID']] = persona_kw

                        sess_id = sess['sessionID']  
                        for utt in sess['dialog']:                    
                            row = {
                                "filename" : filename,
                                "sess_id" : sess_id,
                                "speaker" : utt['speaker'],
                                "speaker_id" : utt['personaID'],
                                "utter" : utt['utterance'],
                            }
                            df = df.append(row, ignore_index=True)
            save_parquet(df, pjoin(dtype_dir_p, sess_lv))
            
    # Save Persona
    with open(pjoin(args.input_folder, "train_persona.json"), 'w') as of:
        json.dump(train_persona, of)

    with open(pjoin(args.input_folder, "valid_persona.json"), 'w') as of:
        json.dump(valid_persona, of)
    return train_persona, valid_persona
            
def preprocess_overlap(args):
    proc_df = pd.DataFrame()
    for dir_type in ['train', 'valid']:
        
        for path in list(iglob(pjoin(args.origin_folder, f"{dir_type}/**.parquet"))):
            proc_df = pd.DataFrame()
            sess_data = pd.read_parquet(path)
            
            # 각 Session id 별, 한 발화자가 연속적으로 발화하는 경우 처리
            for sess_id in tqdm(sess_data['sess_id'].unique(), desc=path):
                sess = sess_data[sess_data['sess_id'] == sess_id]

                i = 0
                while i < len(sess):
                    cur_row = sess.iloc[i]
                    cur_utt = cur_row['utter']

                    i += 1
                    while i < len(sess) and sess.iloc[i]['speaker_id'] == cur_row['speaker_id']:
                        try:
                            cur_utt += "" + sess.iloc[i]['utter']
                        except Exception:
                            pass
                        i += 1

                    row = {
                        'sess_id' : sess_id,
                        'speaker' : cur_row['speaker'],
                        'speaker_id' : cur_row['speaker_id'],
                        'utter' : cur_utt,
                    }
                    proc_df = proc_df.append(row, ignore_index=True)
            
            new_path = path.replace('origin', 'proc')  
            try:
                save_parquet(proc_df, new_path.split(".")[0])
            except:
                proc_df.to_csv(new_path.split(".")[0] + ".csv", index=False)
                
def build_with_persona(args, train_persona, valid_persona):
    # Persona를 포함한 Preprocessed dataset 구축
    for d_type in list(iglob(pjoin(args.proc_folder, "**"))):    
        for path in iglob(pjoin(d_type, "**.parquet")):
            data = pd.read_parquet(path)
            
            pre_contexts = []
            for s_id in data['sess_id']:
                if 'train' in d_type:
                    pre_contexts += [train_persona[s_id]]
                else:
                    pre_contexts += [valid_persona[s_id]]
            data['prefix'] = pre_contexts
            save_parquet(data, path.replace(".parquet", ""))
            

def make_turn(sub_data : pd.DataFrame):    
    query = sub_data.iloc[0]['utter'] 
    query = query if query is not None else " "
    k = len(sub_data) - 1 if sub_data.iloc[-1]['speaker'] == 'speaker1' else len(sub_data) - 2
    for i in range(1, k):
        utt = " " if sub_data.iloc[i]['utter'] is None else sub_data.iloc[i]['utter']
        query += DELIMITER + utt
    
    # speaker1의 발화를 모델링하기 때문에, speaker1의 마지막 발화를 reply로 설정
    reply = "" if sub_data.iloc[k]['utter'] is None else sub_data.iloc[k]['utter']
    return query, reply

def build_multiturn_dataset(args):
    # Persona를 포함한 Preprocessed dataset 구축
    for d_type in list(iglob(pjoin(args.proc_folder, "**"))): 
        mt_data = pd.DataFrame() 
        for path in iglob(pjoin(d_type, "**.parquet")):
            data = pd.read_parquet(path).dropna()
        
            for sess_id in tqdm(data['sess_id'].unique(), desc=f'{path}'):
                sess = data[data['sess_id'] == sess_id]

                # Prefix에 prevTimeInfo는 배제 (speaker1의 페르소나만 포함)
                prefix = sess.iloc[0]['prefix'].split(DELIMITER)[0]
                query, reply = make_turn(sess)
                row = {
                    'sess_id': sess_id,
                    'query' :  prefix + DELIMITER + query,
                    'reply' : reply,
                }
                mt_data = mt_data.append(row, ignore_index=True)
        filename = d_type.split("/")[-1].split(".")[0]
        save_parquet(mt_data, pjoin(args.output_folder, f"{filename}"))
        
def preprocess_dataset(args):
    mkdir_p(args.proc_folder)
    mkdir_p(args.origin_folder)
    mkdir_p(args.output_folder)

    # json 파일을 Dataframe 형태로 변환
    # 각 발화자의 페르소나 추출
    train_persona, valid_persona = json_to_parquet(args)
    
    # 한 발화자의 연속 발화 전처리
    preprocess_overlap(args)
    
    # 페르소나를 포함한 데이터셋 구축
    build_with_persona(args, train_persona, valid_persona)
    
    # 멀티턴 데이터셋 구축 (k개의 턴을 사용)

    build_multiturn_dataset(args)