import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset.MutiCls import MutiCls


class MyDataModule(pl.LightningDataModule):
    
    def __init__(self, params):

        super().__init__()
        self.data_dir = params['data_path']
        self.batch_size = params['batch_size']
        
        # 调用dm.size()返回self.dims
        # 设置默认维度
        # 可以在dm.setup()中动态分配
        self.dims = params['input_dim']
        self.num_classes = params['num_classes']

        self.params = params
        
    def prepare_data(self):
        # downloade
        pass

    def setup(self, stage=None):

        # 指定在数据加载器中使用的训练/验证数据集
        if stage == 'fit' or stage is None:
            muticls_train = MutiCls(self.data_dir, params = self.params, train=True)
            muticls_val = MutiCls(self.data_dir, params = self.params, train=False)
            
            self.mnist_train = muticls_train
            self.mnist_val = muticls_val

        # 指定测试数据集用于dataloader
        if stage == 'test' or stage is None:
            self.mnist_test = MutiCls(self.data_dir, params = self.params, train=False)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size = self.batch_size)

    def test_dataloader(self):
        pass