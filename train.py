import os
import sys

from torchvision import transforms
import pytorch_lightning as pl 
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
sys.path.append('../')
from data.data import MyDataModule as Helmet
from utils.trainer import Classify
from utils.utils import get_yaml

if __name__ == "__main__":
    
    seed_everything(7)
    
    # params

    YAML_FILE = './config/classify.yaml'
    params = get_yaml(YAML_FILE)
    NUM_WORKERS = int(os.cpu_count() / 2)
    params['num_workers'] = NUM_WORKERS

    # dataset 
    transform_train = transforms.Compose([
            
            transforms.RandomChoice ([transforms.Pad(50, fill=0, padding_mode='constant'),
                                    transforms.Pad(50, fill=0, padding_mode='edge'),
                                    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                                    transforms.RandomRotation(30)
                                    ]),
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),  # 归一化，像素值除以255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正规化，像素分布转换为同分布。这里的mean、std是imagenet上的均值标准差。
        ])

    transform_val = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),  # 归一化，像素值除以255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正规化，像素分布转换为同分布。这里的mean、std是imagenet上的均值标准差。
        ])

    
    dm = Helmet(data_dir = params['data_path'], 
                transform = [transform_train, transform_val], 
                dims = params['input_dim'], 
                num_classes = params['num_classes'],
                batch_size = params['batch_size'])

    # model

    model = Classify(hparams = params)
    logger = TensorBoardLogger(params['save_path'] + 'tb_logs', name='')

    trainer = pl.Trainer(
        max_epochs=params['max_epoch'], 
        accelerator=params['accelerator'], 
        gpus=params['num_gpus'],
        weights_save_path=params['save_path'] + 'weights/',
        logger=logger,
        precision=params['precision'],
        limit_train_batches=1.0,
        enable_checkpointing=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        val_check_interval=1.0,
        max_steps=-1,
        strategy="ddp_find_unused_parameters_false"
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
