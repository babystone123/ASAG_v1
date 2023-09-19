# ASAG_v1
该项目为自动简答题评分代码
# 项目文件介绍
1、automatic_grading_model.py 为主要模型文件，该文件负责数据集类创建、分批操作、模型创建、模型训练和验证；
2、documents_preprocess.py 负责数据预处理，主要是加载数据集，然后返回符合模型训练的形式；
3、model_judge.py 该文件的作用是评估模型，接收模型预测值和标签，输出对应评价指标。
# 运行准备
模型使用的数据的为公开数据集，地址：https://web.eecs.umich.edu/~mihalcea/downloads.html#saga

下载依赖后，修改三个文件中个种地址即可，应当修改的地址有：数据集地址、模型地址（如果采用本地加载的话）、数据储存地址、模型储存和读取地址等

# 备注
automatic_grading_model.py文件中还包含了其他功能，如损失的存储和绘制、文本压缩、学习率衰减等，可选择使用。
