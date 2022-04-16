
# 实验二说明

## 运行环境

AI Studio (2核CPU，总内存8GB，总硬盘100GB，Python3.7.4)

## 编译运行方式

使用Python中的以下库：

* os
* copy
* math
* time
* numpy
* codecs
* random
* gensim (word2vec)

## 文件目录（开发端）

以下是在AI Studio平台上开发时的文件目录。源程序中的路径设置与文件目录结构相关联。
```
.
└── aistudio
    ├── dev.txt         // 验证集
    ├── entity_with_text.txt      // 实体描述
    ├── relation_with_text.txt    // 关系描述
    ├── src
    │   ├── method1
    │   │   └── TransE.ipynb      // 源程序
    │   └── method2
    │       └── Word2vec.ipynb
    ├── test.txt        // 测试集
    └── train.txt       // 训练集
```

## 关键函数

`TransE.py`中：
* `data_loader`函数读入训练/验证/测试数据集（对应于`status`参数值为$0/1/2$的情况），返回实体集、关系集和三元组集。`transE_loader`函数完成模型参数的读入。
* `distance`函数计算向量的距离。
* `TransE`类中定义的方法说明：
  + `init`方法指定TransE模型的各种参数与数据源。
  + `emb_initialize`方法为每个实体和关系初始化向量，并归一化。
  + `train`方法对模型进行训练，每次从训练集中随机选择`batch_size`个三元组，并随机构成一个错误的三元组$S'$（`Corrupt`方法）进行更新（`update_embeddings`方法）。
* `eva`函数完成模型的验证或预测，取决于`test`的值。对头实体没有在训练集中出现的情况输出编号1到5，否则找距离最小的5个尾实体编号，与验证集给出的尾实体比较或直接输出。在验证结束后输出相应的准确率。
* 主函数依次完成模型训练、模型验证和模型预测。
  
`word2vec.py`依次完成以下操作:
* 读入实体和关系的描述文件，联合训练word2vec模型。
* 计算实体和关系的向量表示。
* 对测试集中的每个（头实体，关系）二元组，计算特定指标最大（或最小，取决于指标的含义）的5个实体进行输出。这里特定指标可以是尾实体和头实体、关系的余弦相似度之和（积），也可以是尾实体和头实体的余弦相似度等。

## 文件内容

在提交的文件中，`src/method1`目录下存放TransE方法的源代码，`src/method2`目录下存放word2vec方法的源代码。`src/submit`目录下存放最佳结果，根目录下存放`README`和实验报告。
