import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor, RichModelSummary
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .net_dict import Model
from .loss_dict import Loss

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

    def forward(self, x):
        return self.model(x)

    def load_model(self, predicted_model_path = ''):
        # print(self.hparams)
        model_name = self.hparams['hparams']['model_name']
        m = Model(model_name)

        m.print_model
        model = m.get_model

        if predicted_model_path == '':
            return model

        pretrained_dict = torch.load(predicted_model_path)['state_dict']
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model.state_dict():
                # print(k, k.split('model.')[-1])
                new_k = k
                new_pretrained_dict[new_k] = v
        model.load_state_dict(new_pretrained_dict)

        return model

    def load_loss(self):

        loss_name = self.hparams['hparams']['loss_name']
        loss = Loss(loss_name)

        return loss
     
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
        return loss, acc

    def training_step_end(self, step_output: STEP_OUTPUT):
        # self.log('train_loss', step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # step_output 是 training_step 的 return值
        loss, acc = step_output
        self.log('train_step_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_step_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def training_epoch_end(self, outputs):
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        # self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True, logger=True)
        losses, accs = outputs
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

        return loss, acc

    def validation_step_end(self, step_output: STEP_OUTPUT):
        # self.log('train_loss', step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # step_output 是 training_step 的 return值
        loss, acc = step_output
        self.log('val_step_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_step_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        # self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True, logger=True)
        losses, accs = outputs
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
            filename='{epoch}-{train_epoch_acc:.2f}-{val_epoch_acc:.2f}',
            save_top_k=-1,
            save_last=True
        )
        model_summary = RichModelSummary(max_depth=-1)  # ModelSummary(max_depth=-1)
        device_stats = DeviceStatsMonitor()
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        return [model_checkpoint, model_summary, lr_monitor, device_stats]
