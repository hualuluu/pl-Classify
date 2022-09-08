import torch

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor, RichModelSummary

from torch.optim.lr_scheduler import ReduceLROnPlateau
from .nets.net import Model
from .loss.loss import Loss
from .optimizer.optimizer import Optimizer

class MutiCls_Classify(pl.LightningModule):
    def __init__(
        self,
        hparams={
            'batch_size': 128,
            'auto_scale_batch_size': True,
            'auto_lr_find': True,
            'learning_rate': 1e-3,
            'reload_dataloaders_every_epoch': False,
            'num_classes': 2,
            'model_name':'resnet18',
            'loss_name':'cross_entropy'
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
        
        self.model = self.load_model()
        self.loss = self.load_loss()
        self.optimizer, self.scheduler = self.load_optimizer()

    def forward(self, x):
        
        return self.model(x)

    def load_model(self):
        # print(self.hparams)
        model_name = self.hparams['hparams']['model_name']
        num_classes = self.hparams['hparams']['num_classes']
        predicted_model_path = self.hparams['hparams']['predicted_model_path']

        m = Model(model_name, num_classes)
        model = m.get_model()

        # model state dict
        if predicted_model_path == '':
            return model

        pretrained_dict = torch.load(predicted_model_path)['state_dict']
        new_pretrained_dict = {}

        for k, v in pretrained_dict.items():
            new_k = k.split('model.')[-1]
            if new_k in model.state_dict():
                new_pretrained_dict[new_k] = v
        model.load_state_dict(new_pretrained_dict)

        return model

    def load_loss(self):

        loss_name = self.hparams['hparams']['loss_name']
        l = Loss(loss_name)
        loss = l.get_loss()
        
        return loss

    def load_optimizer(self):
        optimizer_name = self.hparams['hparams']['optimizer_name']
        scheduler_name = self.hparams['hparams']['scheduler_name']
        lr = self.hparams['hparams']['learning_rate']
        op = Optimizer(optimizer_name, scheduler_name, self.parameters(), lr)
        optimizer, scheduler = op.get_optimizer()
        return optimizer, scheduler

    def accuracy(self, pred, label):
        # currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(pred, -1), label.squeeze()).to(torch.float32)) / len(label)
        return acc

    def training_step(self, batch, batch_idx):
        
        image, label = batch
        pred = self(image)
        loss = self.loss(pred, label.squeeze())
        acc = self.accuracy(pred, label)
        return {
            "loss" :loss, 
            "acc" : acc
            }

    def training_step_end(self, step_output: STEP_OUTPUT):
        # self.log('train_loss', step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # step_output 是 training_step 的 return值
        # print(step_output)
        loss = step_output["loss"]
        acc = step_output["acc"]

        self.log('train_step_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log('train_step_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def training_epoch_end(self, outputs):
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        # self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True, logger=True)
        # print(outputs)
        losses = []
        accs = []
        
        for ou in outputs:
            losses.append(ou['loss'])
            accs.append(ou['acc'])

        self.log('train_epoch_loss', torch.stack(losses).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('train_epoch_acc', torch.stack(accs).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def validation_step(self, batch, batch_idx):
        # we currently return the accuracy as the validation_step/test_step is run on the IPU devices.
        # Outputs from the step functions are sent to the host device, where we calculate the metrics in
        # validation_epoch_end and test_epoch_end for the test_step.

        image, label = batch
        pred = self(image)

        loss = self.loss(pred, label.squeeze())
        acc = self.accuracy(pred, label)

        return {
            "loss" :loss, 
            "acc" : acc
            }

    def validation_step_end(self, step_output: STEP_OUTPUT):
        # self.log('train_loss', step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # step_output 是 training_step 的 return值
        loss = step_output["loss"]
        acc = step_output["acc"]

        # self.log('val_step_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log('val_step_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        # self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True, logger=True)
        
        losses = []
        accs = []
        
        for ou in outputs:
            losses.append(ou['loss'])
            accs.append(ou['acc'])

        self.log('val_epoch_loss', torch.stack(losses).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_epoch_acc', torch.stack(accs).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=False)

    # def test_step(self, batch, batch_idx):
    
    #     image, label = batch
    #     pred = self(image)
    #     acc = self.accuracy(pred, label)
    #     # 不建议放在这里，放到end_step比较好
    #     # self.log("test_acc", torch.stack(acc).mean(), prog_bar=True)
    #     return acc

    # def test_epoch_end(self, outputs):
        
    #     self.log("test_acc", torch.stack(outputs).mean(), prog_bar=True, logger=False)

    def configure_optimizers(self):
       
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_epoch_loss",
                "frequency": 1  # "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }
    
    def configure_callbacks(self):
        save_path = self.hparams['hparams']['save_path']
        model_checkpoint = ModelCheckpoint(
            dirpath=save_path,
            monitor='val_epoch_acc',
            filename='{epoch}-{train_epoch_acc:.2f}-{val_epoch_acc:.2f}',
            save_top_k=-1,
            save_last=True
        )
        model_summary = RichModelSummary(max_depth=-1)  # ModelSummary(max_depth=-1)
        device_stats = DeviceStatsMonitor()
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        return [model_checkpoint, model_summary, lr_monitor, device_stats]

