from torch.utils.data import DataLoader, random_split
from data.dataset.MutiCls import MutiCls
import pytorch_lightning as pl

class MyDataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir = '',transform = [None, None], dims = (256, 256), num_classes = 2,batch_size = 128):

        super().__init__()
        self.data_dir = data_dir
        self.transform_train, self.transform_val = transform
        self.batch_size = batch_size
        
        # 调用dm.size()返回self.dims
        # 设置默认维度
        # 可以在dm.setup()中动态分配
        self.dims = dims
        self.num_classes = num_classes

    def prepare_data(self):
        # downloade
        pass

    def setup(self, stage=None):

        # 指定在数据加载器中使用的训练/验证数据集
        if stage == 'fit' or stage is None:
            muticls_train = MutiCls(self.data_dir, train=True, transform=self.transform_train)
            muticls_val = MutiCls(self.data_dir, train=False, transform=self.transform_val)
            
            self.mnist_train = muticls_train
            self.mnist_val = muticls_val

        # 指定测试数据集用于dataloader
        if stage == 'test' or stage is None:
            self.mnist_test = MutiCls(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size = self.batch_size)

    def test_dataloader(self):
        pass