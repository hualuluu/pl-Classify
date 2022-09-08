import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
class Optimizer():
    def __init__(self, optimizer_name, scheduler_name, model_params, lr):

        if optimizer_name == 'adm':
            self.optimizer = torch.optim.Adam(
                            model_params, 
                            lr
            )

        if scheduler_name == 'ReduceLROnPlateau':
            # https://www.cnblogs.com/emperinter/p/14170243.html
            self.scheduler = ReduceLROnPlateau(
                    self.optimizer, 
                    factor=0.6, 
                    patience=2, 
                    verbose=True, 
                    mode="min", 
                    threshold=1e-3, 
                    min_lr=1e-8, 
                    eps=1e-8
                )
    
    def get_optimizer(self):    
        
        return self.optimizer, self.scheduler
