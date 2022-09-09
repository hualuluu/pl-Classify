class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer) #用于提取特征图与梯度图
    def generate_cam(self, input_image, target_class=None):
        #1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        #one hot编码，令目标类置1
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
        target = conv_output.data.numpy()[0]
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.model.zero_grad()
        # 步骤1.2.2 计算反向传播
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # 步骤1.2.3 获取目标层各特征图的梯度
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # 初始化热力图
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # 步骤2.2 计算各特征图的加权值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        #步骤2.3 对热力图进行后处理，即将结果变换到0~255
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam