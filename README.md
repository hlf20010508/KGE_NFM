本代码使用yamanishi_08数据集，其他数据集请自行处理

进入data文件夹运行对应数据集的.py文件来下载数据集

进入eg_model文件夹运行eg_model.py来下载kge模型

<br/>

运行环境

kge_nfm.py kge_rf.py:

python 3.7

Tensorflow 1.15.0

cuda 10.0

<br/>

deepdti.py:

Pytorch 1.11.0

cuda 11.3

<br/>

可以使用pyenv管理python版本，使用pipenv创建虚拟环境

<br/>

创建虚拟环境

在当前项目根目录下运行命令

```
pipenv --python 3.7
```

安装requirements.txt中的依赖
```
pipenv run pip install -r requirements.txt
```

安装rdkit拓展
```
pipenv run pip install git+https://github.com/bp-kelley/descriptastorus
```
<br/>

使用train_all.py一键运行
```
pipenv run python train_all.py
```

<br/>

程序运行日志保存在logs文件夹

输出结果保存在output文件夹，包含曲线上的点、auc以及模型
