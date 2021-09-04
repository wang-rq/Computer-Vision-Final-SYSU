# README

### 文件结构及文件说明

```
18340166-王若琪-计算机视觉期末作业
    ├──result [结果]
    │  	├── 1 [作业一结果]
    │  	│  	├─gif [两张gif动图]
    │  	│  	└─others [测试集十张图片的最终效果图]
    │  	├── 2 [作业二结果]
    |   |	├──result_color [随机颜色的分割结果展示]
    │  	│  	└──result_gt_seg [经过区域标记后的结果展示]
    │  	│  	└──result.png [求得的iou结果展示图]
    │  	└── 3 [作业三结果]
    |   	├──test_features [生成的测试集特征及标签]
    │  	  	└──train_features [生成的训练集特征及标签]
    │  	  	└──result.png [求得的iou结果展示图]
    └──src [源代码]
    │   ├──1 [作业一源码]
    │   ├──2 [作业二源码]
    │   └──3 [作业三源码]
    └──README.md
    │
    └──report.pdf [实验报告（包含全部作业 1~3 的内容）]
```

### 代码运行测试方法

1. 作业一：到 src/1/ 目录下，先在 config.py 中设置好文件路径，再在终端输入：

   ```
   python main.py
   ```

2. 作业二：到 src/2/ 目录下，先在 config.py 中设置好文件路径，再在终端输入：

   ```
   python main.py
   ```

3. 作业三：到 src/3/ 目录下，先在 config.py 中设置好文件路径，再在终端输入：

   ```
   python main.py
   ```