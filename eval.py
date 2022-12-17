
# -*- coding: utf-8 -*-

import re, json
import torch
import pandas as pd
from iglob import iglob
from os.path import join as pjoin

from dataloader import DELIMITER
from utils.data_utils import encode
from utils.model_utils import S_TKN, U_TKN

from utils.morp_utils import *
from auto_regressive_model import AutoRegressiveModel
from seq2seq_model import Seq2SeqModel

special = re.compile('[^\sA-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ@?!~,.ᄒ><\^+]')
doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\D)\\1{2,}')

'''
Description
-----------
생성된 시퀀스 후처리 함수
'''
def repeat_normalize(sent, num_repeats=2):
    sent = special.sub('',sent)
    if num_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()

def proc_reply(reply):
    proc_text = re.sub('(<pad>|<unk>|<u>)', '', reply)
    proc_text = repeat_normalize(proc_text, num_repeats=3)
    return proc_text


'''
Description
-----------
k개의 발화를 사용하여 input context를 생성하는 함수
'''
def make_query(dialog, k=2):
    query = dialog[-k]
    for utt in dialog[-k + 1:]:
        query += DELIMITER + utt['utterance']
    return query

'''
Description
-----------
Autoregressive Model을 이용한 reply 생성 함수
'''
def reply_ar(args, model, tokenizer, device, query):
    u_tkn, s_tkn = U_TKN, S_TKN
            
    sys_response = ''

    # encodinig user utterance
    q_toked = tokenizer.tokenize(u_tkn + query)
    if len(q_toked) >= args.max_len:
        q_toked = [q_toked[0]] + q_toked[-(int(args.max_len/2))+1:]

    # inference
    for iter_ in range(args.max_len):
        a_toked = tokenizer.tokenize(s_tkn + sys_response)
        token_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(\
            q_toked + a_toked)).to(device=device)

        logits = model(token_ids)
        gen = tokenizer.convert_ids_to_tokens(torch.argmax(logits, \
            dim=-1).squeeze().cpu().tolist())[-1]
        if gen == tokenizer.eos_token:
            break
        sys_response += gen.replace('▁', ' ')

    sys_response= sys_response.strip()
    reply = proc_reply(sys_response)

    return reply
    

'''
Description
-----------
Sequence-to-Sequence Model을 이용한 reply 생성 함수
'''
def reply_s2s(args, model, tokenizer, device, query):            
    # encoding user utterance
    enc_input, attention_mask = encode(tokenizer=tokenizer, \
        sent=tokenizer.bos_token+query+tokenizer.eos_token, \
        max_len=args.max_len)

    enc_input = torch.LongTensor(enc_input).unsqueeze(0).to(device=device)
    attention_mask = torch.FloatTensor(attention_mask).unsqueeze(0).to(device=device)

    sys_response = ''

    # inference
    for iter_ in range(args.max_len-1):
        dec_input, dec_attention_mask = encode(tokenizer=tokenizer, \
            sent=tokenizer.bos_token+sys_response, max_len=args.max_len)

        dec_input = torch.LongTensor(dec_input).unsqueeze(0).to(device=device)
        dec_attention_mask = torch.FloatTensor(dec_attention_mask).unsqueeze(0).to(device=device)
    
        inputs = {
            "input_ids": enc_input,
            "attention_mask" : attention_mask,
            "decoder_input_ids" : dec_input,
            "decoder_attention_mask" : dec_attention_mask,
            "labels": None
        }
        outs = model(inputs)
        gen = tokenizer.convert_ids_to_tokens(torch.argmax(outs.logits,\
            dim=-1).squeeze().cpu().tolist())[-1]
        if gen == tokenizer.eos_token:
            break
        sys_response += gen.replace('▁', ' ')

    sys_response = sys_response.strip()
    reply = proc_reply(sys_response)
            
    return reply
    
    
'''
Description
-----------
GPT2 기반 대화 모델 test data에서의 테스트
'''
def eval_model(args, model, device):
    tokenizer = model.tokenizer
    
    with torch.no_grad():
        
        for path in iglob(pjoin(args.input_folder, "**.json")):
            with open(path, encoding="UTF-8") as f:
                entire_info = json.load(f)
                session = entire_info['sessionInfo'][0]
                    
                # Get prev-session information
                # prev_time = session['prevTimeInfo']['timeNum'] + session['prevTimeInfo']['timeUnit']
                persona = entire_info['personaInfo']['clInfo']['personaFeatures']
                persona_kw = ' '.join(extract_keyword(persona))
                prev_info = persona_kw
                    
                result = pd.DataFrame(columns=['sent_type', 'sent', '이전 세션 정보 사용', '적절한 발화 여부'])
                # Append Speaker Persona Summary
                result = result.append({
                    'sent_type': 'speaker_1_summary',
                    'sent': '\n'.join(entire_info['prevAggregatedpersonaSummary']['speaker1']),
                }, ignore_index=True)
                result = result.append({
                    'sent_type': 'speaker_2_summary',
                    'sent': '\n'.join(entire_info['prevAggregatedpersonaSummary']['speaker2']),
                }, ignore_index=True)
                    
                # Append Dialogue
                for utt in session['dialog']:
                    result = result.append({
                        'sent_type' : utt['speaker'],
                        'sent' : utt['utterance'],
                    }, ignore_index=True)
                    
                # Make Context
                context = prev_info + DELIMITER + make_query(session, k=args.k)
                if args.model_type == 'gpt2':
                    reply = reply_ar(args, model, tokenizer, device, context)
                else:
                    reply = reply_s2s(args, model, tokenizer, device, context)
                        
                result = result.append({
                    'sent_type': 'speaker_1_generated',
                    'sent': reply,
                }, ignore_index=True)

                filename = entire_info['FileInfo']['filename'].replace("json", "xlsx")
                result.to_excel(pjoin(args.output_folder, filename), index=False, encoding='utf-8')
           
           
def evaluation(args, **kwargs):
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    # load checkpoint
    if args.model_pt is not None:
        if args.model_type == 'gpt2':
            model = AutoRegressiveModel.load_from_checkpoint(\
                checkpoint_path=args.model_pt, hparams=args, device=torch.device(device))
        else:
            model = Seq2SeqModel.load_from_checkpoint(\
                checkpoint_path=args.model_pt, hparams=args, device=torch.device(device))

    model = model.cuda() 

    # freeze model params   
    model.eval()
    model.freeze()

    eval_model(args, model, device)
    