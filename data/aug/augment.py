import cv2
import numpy as np
import random

from math import *

class Augment():
    def __init__(self, params):
        self.params = params
        self.input_dims = self.params['input_dim']

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_augment(self, image, train = True):
        image = np.array(image, dtype = np.float32)[:, :, ::-1]
        self.image_h, self.image_w, _ = image.shape
        self.input_h, self.input_w = self.input_dims
        if train:
            if self.params['HorizontalFlip']:
                image = self.horizontalflip(image)
            if self.params['VerticalFlip']:
                image = self.verticalflip(image)
            if self.params['RandomRotate']:
                image = self.randomrotate(image)
            if self.params['RandomContrast']:
                image = self.randomcontrast(image)
            if self.params['RandomLight']:
                image = self.randomlightness(image)
            if self.params['RandomChannel']:
                image = self.randomchannel(image)
            if self.params['RandomResize']:
                image = self.randomresize(image)
            if self.params['letterbox']:
                image = self.letterbox(image)
            else:
                image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            if self.params['Normalize']:
                mean = self.params['mean']
                std = self.params['std']
                
                image = (image / 255.0 - mean) / std
        else:
            if self.params['letterbox']:
                image = self.letterbox(image)
            else:
                image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            
            if self.params['Normalize']:
                mean = self.params['mean']
                std = self.params['std']
                image = (image / 255.0 - mean) / std

        return image

    def randomrotate(self, image):
        degree = self.rand(0, 360)
        M = cv2.getRotationMatrix2D((self.image_w / 2,self.image_h / 2), degree, 1)
        nh = int(self.image_w * fabs(sin(radians(degree))) + self.image_h * fabs(cos(radians(degree))))
        nw = int(self.image_h * fabs(sin(radians(degree))) + self.image_w * fabs(cos(radians(degree))))
        
        M[0, 2] += (nw - self.image_w) / 2  
        M[1, 2] += (nh - self.image_h) / 2 

        image = cv2.warpAffine(image, M, (nw, nh), borderValue = (128, 128, 128))

        return image

    def randomcontrast(self, image):
        alpha = self.rand(0.5, 2) 
        image = image * alpha
        image = np.clip(image, 0, 255)
        return image

    def randomlightness(self, image):

        bias = random.uniform(-50, 50)  
        image = image + bias
        image = np.clip(image, 0, 255)
        return image

    def randomchannel(self, image):
        p = self.rand(0, 1)
        # image = BGR --OPENCV INPUT
        if p < 0.2:
            # remove red
            image[:,:,-1] = 0
        elif p < 0.4:
            # remove green
            image[:,:,1] = 0
        elif p < 0.6:
            # remove blue
            image[:,:,0] = 0
        elif p < 0.8:
            # tran channel
            order = np.random.permutation(3)
            image = image[:,:,order] 

        return image
    
    def horizontalflip(self, image):
        return cv2.flip(image, 1)
    
    def verticalflip(self, image):
        return cv2.flip(image, 0)

    def randomresize(self, image):
        resize_jitter = .3
        # 随机对原始图片进行resize, 并进行长宽比的扭曲
        new_rate = self.rand(1 - resize_jitter,1 + resize_jitter)
        
        scale = self.rand(.25, 2)
        
        self.image_w = int(scale * self.image_w)
        self.image_h = int(self.image_w * new_rate)
        
        image   = cv2.resize(image, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)

        return image

    def letterbox(self, image):
        scale   = min(self.input_w / self.image_w, self.input_h / self.image_h)
        nw      = int(self.image_w * scale)
        nh      = int(self.image_h * scale)
        
        image   = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # print(self.input_h - nh, (self.input_h - nh) / 2)
        deta_h = int((self.input_h - nh) / 2)
        deta_w = int((self.input_w - nw) / 2)
        
        top = deta_h
        bottom = deta_h
        left = deta_w
        right = deta_w

        if deta_h * 2 != self.input_h - nh:
            top = deta_h + 1
        if deta_w * 2 != self.input_w - nw:
            left = deta_w + 1

        new_image = cv2.copyMakeBorder( image,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        cv2.BORDER_CONSTANT, 
                                        value=(128,128,128)
        )
        # print(new_image)
        return new_image

if __name__ == "__main__":
    
    
    yaml_file = '../../config/classify.yaml'
    import sys 
    sys.path.append('../../')
    from utils.utils import get_yaml
    params = get_yaml(yaml_file)

    au = Augment(params)

    image_dict = ['/media/worker/98f56e45-a9c8-4f4e-b3f1-08a2d16a7ec1/liliang/Project/Classify/datasets/helmet/head/55d73c53-96a7-491a-ac7a-2684c1bffdb2_1118_171208_0.jpg',
    './input.jpg',
    '/media/worker/98f56e45-a9c8-4f4e-b3f1-08a2d16a7ec1/liliang/Project/Classify/datasets/helmet/helmet/9c6e272b7f799952176add4d9a946751_0.jpg'
    ]
    image = cv2.imread(image_dict[1])
    image = au.get_augment(image)
    print(image.shape)
    cv2.imwrite('./save.jpg', image)