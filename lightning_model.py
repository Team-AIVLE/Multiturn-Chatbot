import argparse

from pytorch_lightning.core.lightning import LightningModule

class LightningModel(LightningModule):
    """Super class"""
    def __init__(self, hparams, device='cuda'):
        super(LightningModel, self).__init__()
        self.hparams = hparams
        self.model_type = hparams.model_type.lower()

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size',
                            type=int,
                            default=4,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        pass

