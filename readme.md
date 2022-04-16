
# 实验三说明

## 运行环境

笔记本电脑1台，Python3.9，VSCode
处理器：AMD Ryzen 7 4800H with Radeon Graphics, 2.90 GHz
机带RAM：16.0GB（15.4GB可用）

## 编译运行方式

直接运行`src/UserCF/UserCF.py`即可，该程序中使用了Python中的`math`和`pathlib`库。

## 关键函数

`src/UserCF/UserCF.py`中：
* `__init__`函数初始化以下参数为空字典：训练集、测试集（本实验中未使用）、相似度函数、每个用户的平均评分和相似度公式中的被除项。
* `initData`函数读入训练集并进行预处理计算。
* `userSimilarity_3`函数计算用户相似度（删去了`userSimilarity_1`和`userSimilarity_2`函数，并对原来的代码进行了覆写）。
* `recommend`函数根据评分预测从高到低，返回Top-100的结果。
* `solv`函数对每个用户调用`recommend`函数，完成结果的输出。
  
## 文件内容

在提交的文件中，`src/UserCF`目录下存放基于用户的协同过滤方法的源代码，`submit`目录下存放最佳结果。根目录下存放数据集`DoubanMusic.txt`、Python环境依赖`requirements.txt`、实验说明文件`readme.md`和实验报告`report.pdf`。
