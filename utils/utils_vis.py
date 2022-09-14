import cv2
from .CAM.GradCam import GradCam
from torchvision import transforms

def get_cam(image, trans, model, layer_name, savepath):
    """
    功能：获得特征热力图,使用的是Grad CAM
    """
    
    trans_image = trans(image).unsqueeze(0)

    cam = GradCam(model, layer_name)
    c = cam.generate_cam(trans_image)

    # import matplotlib.pyplot as plt
    # plt.imshow(c, cmap=plt.cm.jet)
    # plt.savefig('./cam.jpg')

    c = cv2.applyColorMap(cv2.normalize(c , None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
    cv2.imwrite(savepath + 'cam.jpg', c)
    print(type(image))
    add_image = cv2.addWeighted(image, 0.5, c, 0.5, gamma = 0)
    cv2.imwrite(savepath + 'cam_add_image.jpg', add_image)
