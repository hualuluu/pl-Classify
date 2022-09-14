import yaml
import os

def get_singleclass_txt(txtpath):
    image_name = []
    label = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            label.append(int(line.split('|')[1]))
            image_name.append(line.split('|')[0])
    return image_name, label

def get_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

