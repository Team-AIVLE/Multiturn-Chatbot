import torch

from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from lightning_model import LightningModel
from utils.model_utils import load_model
from dataloader import AutoRegressionChatData

'''
Description
-----------
GPT2 기반 대화 모델

huggingface에 공개된 한국어 사전학습 모델 KoGPT2 \
    skt/kogpt2-base-v2 사용
'''
class AutoRegressiveModel(LightningModel):
    def __init__(self, hparams, device='cuda'):
        super(AutoRegressiveModel, self).__init__(hparams, device)
        self.hparams = hparams
        self.neg = -1e18

        self.model_type = hparams.model_type.lower()
        self.model, self.tokenizer = load_model(self.model_type)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):
        output = self.model(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)

        # assign negative logits 
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def validation_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)

        # assign negative logits 
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        return loss_avg

    def validation_epoch_end(self, outputs):
        avg_losses = []
        for loss_avg in outputs:
            avg_losses.append(loss_avg)
        self.log('val_loss', torch.stack(avg_losses).mean(), prog_bar=True)
    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data_path = f'{self.hparams.data_dir}/train.parquet'
        self.train_set = AutoRegressionChatData(data_path, max_len=self.hparams.max_len, tokenizer=self.tokenizer)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=False, collate_fn=self._collate_fn)
        return train_dataloader
    
    def val_dataloader(self):
        data_path = f'{self.hparams.data_dir}/valid.parquet'
        self.valid_set = AutoRegressionChatData(data_path, max_len=self.hparams.max_len, tokenizer=self.tokenizer)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=False, collate_fn=self._collate_fn)
        return val_dataloader

    def generate(self, input_ids, num_beams=1, no_repeat_ngram_size=3, max_length=128, num_return_sequences=1, early_stopping=False):

        return self.model.generate(

            input_ids,

            num_beams = num_beams,

            no_repeat_ngram_size = no_repeat_ngram_size,

            max_length = max_length,

            num_return_sequences = num_return_sequences,

            early_stopping = early_stopping,
        )
