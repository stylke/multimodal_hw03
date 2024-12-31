# 多模态学习2024年第三次作业

## 主题
使用RNN/LSTM/Transformer实现Image Caption模型 



## 作业要求

1. 作业提供了*不完整*的代码，需要同学们根据要求补全代码，并运行得到相应的结果。

2. 本次作业共包含2个任务，由`3.1-RNN_Captioning`、`3.2-Transformer_Captioning`组成(每个任务都有A、B两个section)，两个任务的分值分别是 **45、55**，总分为**100**分。

## 项目文件说明

**注意**：本次作业涉及两套环境（`3.2-Transformer_Captioning`中的`section B`请使用`mml/requirements.txt`，其他section使用`requirements.txt`），因此建议使用conda/venv等virtual environment管理环境
```
├── 3.1-RNN_Captioning.ipynb                   # 任务3-1对应的jupyter notebook
├── 3.2-Transformer_Captioning.ipynb           # 任务3-2对应的jupyter notebook
├── README(本次作业说明).md                      # 本次作业说明
├── mml                                        # 本次作业提供/需要补全的一些文件
│   ├── requirements.txt                       # image caption with clip and LM的所需的一些dependency
│   ├── captioning_solver.py                   # 训练图像描述模型，优化参数，防止过拟合，记录最佳参数及训练历史等模块
│   ├── captioning_solver_transformer.py       # 与captioning_solver.py，主要为任务3-2中transformer做一些适配
│   ├── classifiers                            # 一些基础模型实现的文件夹
│   │   ├── rnn.py                             # RNN和LSTM实现的文件
│   │   └── transformer.py                     # transformer实现的文件
│   ├── coco_utils.py                          # 读取 coco 数据集所需的一些模块
│   ├── data                                   # 实现 image caption dataset所需的文件夹
│   │   ├── __init__.py                        # 显式导入一些模块
│   │   └── dataset.py                         # 实现 image caption dataset(MSCOCO)所需的文件
│   ├── evaluate.py                            # 评测 image caption with clip and LM的文件
│   ├── gradient_check.py                      # 一些梯度检查等模块
│   ├── image_utils.py                         # 一些图像处理的模块
│   ├── model                                  # 实现 caption model with CLIP and LM 所需模型、训练/推理框架
│   │   ├── __init__.py                        # 显式导入一些模块
│   │   ├── model.py                           # 实现 caption model with CLIP and LM 的文件
│   │   └── trainer.py                         # 实现train and test(validation) loop 的文件
│   ├── optim.py                               # 一些optimizer的实现
│   ├── rnn_layers.py                          # rnn/lstm layer实现
│   ├── training.py                            # 训练的launcher
│   ├── transformer_layers.py                  # transformer layer实现
│   └── utils                                  # 一些工具模块实现的文件夹
│       ├── __init__.py                        # 显式导入一些模块
│       ├── config.py                          # 模型配置文件
│       └── lr_warmup.py                       # 学习率设置文件
└── requirements.txt                           # 一些基础的dependency
└── img                                        # framework图片
```

## 作业提交内容

(1) 本项目所有文件包括不限于： 原有以及新增加的`.ipynb`、`.py`、checkpoint、必要的项目运行说明

(2) 请按原始结构提交，保证代码能**重复运行**

(3) 最终的作业请压缩为zip文件，并命名为 名字_学号_multimodal_hw03.zip， 如 `哆摩泰_1800012345_multimodal_hw03.zip`

## 时间要求

本次作业时间为2周，请同学们按时提交。 不接受任何形式的补交。



