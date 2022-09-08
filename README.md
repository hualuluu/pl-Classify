# pl-Classify
使用pytorch-lighting做图像分类训练

### 1.环境安装：  
conda create -n pl python=3.8  
source activate pl  
pip install -r requirement # -i  https://pypi.tuna.tsinghua.edu.cn/simple  


### 2.训练之多类别分类：  
类别之间互斥，标注文件为0,1,2,3....  
#### 2.1 数据准备  
##### 2.1.1 生成训练的txt文件，格式如下：  
* imagepath|cls  
  ```
  image1.jpg|1  
  image2.jpg|0 
  ```
##### 2.1.2 文件层级： 
* 将txt放于datasets文件夹下
```
|--datasets
    |--train.txt
    |--test.txt
```
#### 2.2 文件修改
##### 2.2.1 生成训练的txt文件，格式如下：   
* 修改 config/classify.yaml 文件中  
  data_path：datasets/
  save_path: runs/自己的文件夹  

#### 2.3 文件修改 
```
python train.py
```