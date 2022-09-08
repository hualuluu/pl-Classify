from torch.nn import functional as F

class Loss():
    def __init__(self, loss_name):

        if loss_name == 'cross_entropy':
            loss = F.cross_entropy

        return loss