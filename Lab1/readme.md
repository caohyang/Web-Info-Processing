
# 实验1说明

## 运行环境

笔记本电脑1台，Python3.9，VSCode

## 编译运行方式

使用Python中的以下库：

* os
* nltk
* json
* collections
* sklearn
* pandas
* numpy
* time

## 关键函数

`bool_search.py`中：
* `arr_and`、`arr_or`、`arr_not`函数分别对两个文档编号列表进行交、并、补操作。
* `parse_query`函数按给定优先级处理查询输入，输出记号流。
* `process_query`函数处理记号流：
  + 对查询词项做词根化处理后返回对应的倒排表
  + 对逻辑运算的关键字`AND`、`OR`、`NOT`，返回两边（或单边）列表的运算结果
  + 若查询格式有误，输出报错信息
* 主函数（没有写成函数的形式）完成了以下任务：
  + 遍历数据集中已完成预处理的json文件，建立倒排表
  + 按词项的字典序将倒排表写入表格
  + 调用上述函数完成布尔检索并输出检索结果
  
`semantic_search.py`中依次进行数据的第二次处理（将单词列表连成字符串）、使用`TfidfVectorizer`相关函数计算每个文档的`tf-idf`向量、求解查询的`tf-idf`值。

## 文件内容

src目录下，`pre_processing.py`对原始数据集进行预处理（分词、词根化、去停用词，保留文档编号、标题和正文信息），其余文件内容同实验要求。
