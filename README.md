Evaluation

Face detection evaluation toolkit.
Base on `http://vis-www.cs.umass.edu/fddb/`.

### Results
![](https://github.com/ZhouKai90/Evaluation/blob/master/result.jpg)



### 使用说明

1. 生成针对fddb测试数据的人脸预测结果文件

   在`model_inference/fddb_config.py`中修改相应的参数

   针对不同的模型，生成保存预测结果的*_det.lst文件，例如`model_inference/fddb_mxnet_inference.py`。

2. 生成RCO文件

   修改`evaluation/config.hh`中对应的参数，主要是`fddbDetFile`即步骤一中生成的det文件。

   执行`./evaluation`生成对应的ROC文件。当然很多参数也可以通过命令行参数输入。

   最后生成`xxxDiscROC.txt`和`xxxContROC.txt`文件

3. 生成ROC pdf文件

   将步骤2中得到的`xxxDiscROC.txt`拷贝到`marcopede-face-eval/detections/fddb`下

   根据不同的数据集，运行不同的脚本，例如`python  plot_AP_fddb.py`

   