from torch.nn import functional as F

class Loss():
    def __init__(self, loss_name):

        if loss_name == 'cross_entropy':
            self.loss = F.cross_entropy

    def get_loss(self):    
        
        return self.loss
