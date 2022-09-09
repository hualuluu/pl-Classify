### 训练之多类别分类：  
类别之间互斥，标注文件为0,1,2,3....  
#### 数据准备  
##### 生成训练的txt文件，格式如下：  
* imagepath|cls  
  ```
  image1.jpg|1  
  image2.jpg|0 
  ```
##### 文件层级： 
* 将txt放于datasets文件夹下
```
|--datasets
    |--train.txt
    |--test.txt
```
#### 文件修改
##### 生成训练的txt文件，格式如下：   
* 修改 config/classify.yaml 文件中  
  data_path：datasets/
  save_path: runs/自己的文件夹  

#### 开始训练
```
python train.py
```