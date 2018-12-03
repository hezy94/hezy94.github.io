---
layout: post
title: 'Jupyter Notebook & KDD 2010'
date: 2018-12-03
tags:
---
# Jupyter Notebook & KDD 2010
## Jupyter Notebook安装与使用
1. 安装(注意是要安装pyhton2还是3)

```
#安装jupyter notebokk
pip3 install jupyter

#创建根目录
mkdir jupyter

#生成密钥
python3 -c "import IPython;print IPython.lib.passwd()"
会生成类似sha1:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX的密码

#生成配置文件
jupyter notebook --generate-config

#启动jupyter notebook
root用户：jupyter notebook --allow-root
非root用户：jupyter notebook

#修改配置文件
vi ../.jupyter/jupyter_notebook_config.py
加入修改内容
c.NotebookApp.ip = '*' #允许所有ip访问
c.NotebookApp.allow_root = True
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888 #默认端口
c.NotebookApp.password = u'刚才生成的密文(sha:...)'
c.ContentsManager.root_dir = '/data/jupyter/root' #根目录地址

#打开
jupyter notebook #启动
210.26.49.91:7000 #通过ip加端口访问
```


## [KDD 2010](http://210.26.49.91:7000/notebooks/jiangfan/CNN_Test/DataAnalysis.ipynb)

### 比赛介绍
1. [pslcdatashop](https://pslcdatashop.web.cmu.edu/KDDCup/rules.jsp)
1. [KDD官网](https://www.kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Intro)
 
### 数据集
- 分为了3个Development Data Sets与2个Challenge Data Sets

#### 主要包括了：
- Row：行数
- Anon Student Id：学生匿名ID
- Problem Hierarchy：问题层次
- Problem Name：问题名称
- Problem View：学生需要问题的总次数
- Step Name：步骤名称
- Step Start Time：步骤开始的时间
- First Transaction Time：第一次提交的时候
- Correct Transaction Time：正确步骤产生的时间
- Step End Time：步骤结束的时间（Step Start Time - Step End Time）
- Step Duration (sec)：步骤耗时
- Correct Step Duration (sec)：正确步骤耗时
- Error Step Duration (sec)：错误步骤耗时
- ***Correct First Attempt：***学生步骤第一次正确尝试的概率
- Incorrects：错误尝试的总数
- Hints：提示
- Corrects：对步骤完全正确的尝试次数
- KC(KC Model Name)：知识组成
- Opportunity(KC Model Name)：每次学生与所列知识相遇时+1

#### 评价方式
1. 对Correct First Attempt进行预测
1. 对三个开发集求RMSE，并且对其取均值得到最后结果

#### 其他
1. 一个问题包含了一个或多个步骤，一个步骤由一个或者多个KC组成
1. 使用了问题+步骤作为标识符

### 数据查看

```
import pandas as pd
data = pd.read_csv("../public/algebra_2005_2006/algebra_2005_2006_train.txt",sep='\t')
```

### 数据处理
```
data['id'] = data["Problem Name"] +' '+ data["Step Name"]
data2idx = {c: i for i, c in enumerate(list(set(data['id'])))} 
data['ID'] = [data2idx[c] for c in data['id']]
data.drop(['id'],axis=1,inplace=True)
data
```

### 数据选择
- ID
- Problem View
- Correct Step Duration (sec)
- Hints
- Correct First Attempt

```
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.3)
x_train = train_data.loc[:,['ID','Problem View','Correct Step Duration (sec)','Hints']].fillna(0).values.reshape(len(train_data),1,4,1)
y_train = train_data['Correct First Attempt'].values.reshape(len(train_data),1)

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
```

### 构建卷积层+池化层+全连接层+Softmax层

#### compile
- optimizer的选择
sgd(随机梯度下降)

- loss 使用binary_crossentropy

- 定义了一个RMSE，使用其作为评价标准，另外还提供了accuracy、MSE等

#### fit 
- batch-size的选择
- epoch的选择

## 记录
#### optimizer
1. sgd
loss: 3.7235 - acc: 0.7664 - rmse: 0.2336
1. Adam
loss: 3.7298 - acc: 0.7660 - rmse: 0.2340

## batch-size
1. 128
loss: 3.7158 - acc: 0.7669 - rmse: 0.2331
1. 500
loss: 3.7209 - acc: 0.7666 - rmse: 0.2334
1. 2000
loss: 3.7203 - acc: 0.7666 - rmse: 0.2334
1. 5000
loss: 3.7156 - acc: 0.7669 - rmse: 0.2331
1. 10000
loss: 3.7178 - acc: 0.7668 - rmse: 0.2332

# other
1. 去掉dropout层
loss: 3.7097 - acc: 0.7673 - rmse: 0.2327


