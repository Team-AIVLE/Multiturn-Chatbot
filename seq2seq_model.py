import torch
import logging

from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from lightning_model import LightningModel
from utils.model_utils import load_model
from dataloader import Seq2SeqChatData


'''
Description
-----------
BART 기반 리액션 대화 모델

huggingface에 공개된 한국어 사전학습 모델 KoBART \
    gogamza/kobart-base-v2 사용
'''
class Seq2SeqModel(LightningModel):
    def __init__(self, hparams, device='cuda'):
        super(Seq2SeqModel, self).__init__(hparams, device)
        self.hparams = hparams
        self.model_type = hparams.model_type.lower()
        self.model, self.tokenizer = load_model(hparams.model_type.lower())
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=inputs['decoder_attention_mask'],
                          labels=inputs['labels'], return_dict=True)
    
    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = 2
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        data_path = f'{self.hparams.data_dir}/train.csv'
        self.train_set = Seq2SeqChatData(data_path, max_len=self.hparams.max_len, tokenizer=self.tokenizer)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=False)
        return train_dataloader
 
    def val_dataloader(self):
        data_path = f'{self.hparams.data_dir}/valid.csv'
        self.valid_set = Seq2SeqChatData(data_path, max_len=self.hparams.max_len, tokenizer=self.tokenizer)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=False)
        return val_dataloader
