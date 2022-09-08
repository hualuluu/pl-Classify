import os
import sys
from typing import Any

from torch.nn import functional as F

import torch.nn as nn
import torch
import torchvision 

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor, RichModelSummary
from torch.optim.lr_scheduler import ReduceLROnPlateau

from _DPLUtils.PLMCallback import PLMCallback


import torch.utils.tensorboard.writer as TBWriter


class Classify(pl.LightningModule):
    def __init__(
        self,
        hparams={
            'batch_size': 128,
            'auto_scale_batch_size': True,
            'auto_lr_find': True,
            'learning_rate': 1e-3,
            'reload_dataloaders_every_epoch': False,
            'num_classes': 2
        }
    ):
        '''
            auto_scale_batch_size: 
                # 设置为True，Trainer就会依次尝试用2的幂次方的batch size，直到超出内存
                # 设置为'binsearch'，Trainer会用Binary Search的方式帮你找到最大的Batch Size
                # 注意：如果要用这个功能，在Module里面的__init__()函数中要有:
            auto_lr_find:
                # 可以直接设置为True，Trainer会自动用不同的学习率运行model，然后画出loss和学习率的曲线，帮你找到最合适的学习率
                # 有时候我们会在model中给学习率起其他的名字，比如： self.my_learning_rate = lr
            
        '''
        super().__init__()
        self.save_hyperparameters('hparams')
        # print(f'[MnistInterface]  hparams: {self.hparams}')
        
        self.model = self.create_model()
        # print(f'='*80)
        # TISummary(self.model)
        # print(f'-'*80)
        
    def create_model(self):
        
        # model = torchvision.models.resnet18(pretrained=False, num_classes=3)
        # model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # model.maxpool = torch.nn.Identity()
        model = torchvision.models.mobilenet_v2(pretrained=False)

        # Update Model Structure
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, model.last_channel // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(model.last_channel // 2, 3)
        )
        return model

    def forward(self, x):
        
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.squeeze())
        # 此处相当与对变量cross_entropy进行一次注册，如果不注册，tensorboard不显示。
        # 不建议放在这里，放到end_step比较好
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', value=self.accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        probs = self(x)
        # we currently return the accuracy as the validation_step/test_step is run on the IPU devices.
        # Outputs from the step functions are sent to the host device, where we calculate the metrics in
        # validation_epoch_end and test_epoch_end for the test_step.
        acc = self.accuracy(probs, y)
        # 不建议放在这里，放到end_step比较好
        self.log("val_acc", acc, prog_bar=True, logger=False)
        loss = F.cross_entropy(probs, y.squeeze())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        # 不建议放在这里，放到end_step比较好
        # self.log("test_acc", torch.stack(acc).mean(), prog_bar=True)
        return acc

    def accuracy(self, logits, y):
        # currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y.squeeze()).to(torch.float32)) / len(y)
        return acc

    def training_step_end(self, step_output: STEP_OUTPUT):
        # self.log('train_loss', step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # step_output 是 training_step 的 return值
        self.log('train_loss', step_output, on_step=True, on_epoch=True, prog_bar=True, logger=False)

    def validation_epoch_end(self, outputs):
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        # self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True, logger=True)
        
        self.log('val_loss', torch.stack(outputs).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        

    def test_epoch_end(self, outputs):
        
        self.log("test_acc", torch.stack(outputs).mean(), prog_bar=True, logger=False)

    def configure_optimizers(self):
       
        adam = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams['hparams']['learning_rate'],
        )
        return {
            "optimizer": adam,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    adam, 
                    factor=0.6, 
                    patience=2, 
                    verbose=True, 
                    mode="min", 
                    threshold=1e-3, 
                    min_lr=1e-8, 
                    eps=1e-8
                ),
                "monitor": "val_acc",
                "frequency": 1  # "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }
    
    def configure_callbacks(self):
        
        model_checkpoint = ModelCheckpoint(
            monitor='val_acc',
            filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
            save_top_k=-1,
            save_last=True
        )
        model_summary = RichModelSummary(max_depth=-1)  # ModelSummary(max_depth=-1)
        device_stats = DeviceStatsMonitor()
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        return [model_checkpoint, model_summary, lr_monitor, device_stats]

