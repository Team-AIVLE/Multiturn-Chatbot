
# -*- coding: utf-8 -*-

import re
import torch
import pandas as pd
from os.path import join as pjoin

from dataloader import DELIMITER
from utils.data_utils import encode
from utils.model_utils import S_TKN, U_TKN

from auto_regressive_model import AutoRegressiveModel
from seq2seq_model import Seq2SeqModel

repeatchars_pattern = re.compile('(\D)\\1{2,}')

'''
Description
-----------
생성된 시퀀스 후처리 함수
'''
def repeat_normalize(sent, num_repeats=2):
    if num_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    sent = re.sub('[\s]+', ' ', sent)
    return sent.strip()

def proc_reply(reply):
    proc_text = re.sub('(<pad>|<unk>)', '', reply)
    proc_text = repeat_normalize(proc_text, num_repeats=3)
    return proc_text

'''
Description
-----------
사용자 입력이 유효한지 판단
'''
def is_valid(query):
    if not re.sub('[\s]+', '', query):
        return False
    return True

'''
Description
-----------
GPT2 기반 대화 모델 test data에서의 테스트
'''
def eval_ar(args, model, device, test_data):
    u_tkn, s_tkn = U_TKN, S_TKN
    tokenizer = model.tokenizer

    sys_responses = replies = [], []
    with torch.no_grad():
        for d in test_data.iterrows():
            row = d[1]
            query = row['proc_query']

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
            print("Sys Response: {}".format(sys_response))

            reply = proc_reply(sys_response)
            
            sys_responses.append(sys_response)
            replies.append(reply)

        # save test result to <save_dir>
        test_data['system_response'] = sys_responses
        test_data['gen_reply'] = replies
        test_data.to_csv(f'{args.save_dir}/{args.model_name}.csv', index=False)
    
'''
Description
-----------
BART 기반 대화 모델 test data에서의 테스트
'''
def eval_s2s(args, model, device, test_data):
    tokenizer = model.tokenizer

    sys_responses = replies = [], []
    with torch.no_grad():
        for d in test_data.iterrows():
            row = d[1]
            query = row['proc_query']

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
            
            sys_responses.append(sys_response)
            replies.append(reply)

            print("User Utterance: {}".format(query))
            print("Reply: {}".format(reply))
        
        # save test result to <save_dir>
        test_data['system_response'] = sys_responses
        test_data['gen_reply'] = replies
        test_data.to_csv(f'{args.save_dir}/{args.model_name}.csv', index=False)
    
'''
Description
-----------
GPT2 기반 대화 모델 사용자 입력에 대한 테스트
'''
def chat_ar(args, model, device):
    u_tkn, s_tkn = U_TKN, S_TKN
    tokenizer = model.tokenizer

    query = input('User Utterance: ')
    with torch.no_grad():
        while is_valid(query):
            sys_response = ''

            # encoding user utterance
            q_toked = tokenizer.tokenize(u_tkn + query)
            if len(q_toked) >= args.max_len:
                query_toked = query_toked[-(int(args.max_len/2)):]

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
            sys_response = sys_response.strip()


            reply = proc_reply(sys_response)
            print("Reply: {}".format(reply))

            query = input('User Utterance: ')

'''
Description
-----------
BART 기반 대화 모델 사용자 입력에 대한 테스트
'''
def chat_s2s(args, model, device):
    tokenizer = model.tokenizer

    query = input('User Utterance: ')
    with torch.no_grad():
        while is_valid(query):
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
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(outs.logits, \
                    dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break
                sys_response += gen.replace('▁', ' ')
            sys_response = sys_response.strip()

            reply = proc_reply(sys_response)
            print("Reply: {}".format(reply))
            
            query = input('User Utterance: ')
           
           
           
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

    # load test dataset
    test_data = pd.read_csv(pjoin(args.data_dir, 'test.csv'))
    test_data = test_data.dropna(axis=0)

    if args.model_type == 'bart':
        if args.chat:
            chat_s2s(args, model, device)
        else:
            eval_s2s(args, model, device, test_data)
    else:
        if args.chat:
            chat_ar(args, model, device)
        else:
            eval_ar(args, model, device, test_data)
