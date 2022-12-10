from transformers import (GPT2LMHeadModel, 
                        PreTrainedTokenizerFast,
                        BartForConditionalGeneration)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

'''
Description
-----------
모델 유형에 따라 사전 학습된 모델과 토크나이저 반환

Input:
------
    model_type: 모델 유형 ('gpt2' or 'bart')
'''
def load_model(model_type):
    if 'bart' == model_type:
        
        model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        return model, tokenizer

    elif 'gpt2' == model_type:
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
        return model, tokenizer

    raise NotImplementedError('Unknown model')