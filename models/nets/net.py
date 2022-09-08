from .ResNet.resnet import ResNet18

class Model():
    def __init__(self, model_name, num_classes):

        if model_name == 'resnet18':
            self.model = ResNet18(num_classes)

    def get_model(self):    
        
        return self.model
    
    def print_model(self):

        print(self.model)