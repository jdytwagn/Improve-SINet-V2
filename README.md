# Improve-SINet-V2
基于深度学习的伪装目标检测网络模型设计及实现
# 数据集
采用主流的四个数据集COD10K、CAMO、CHAMELEON 和 NC4K，数据集下载链接（百度网盘版）：通过网盘分享的文件：Dataset
链接: https://pan.baidu.com/s/159VrPMukt1Zii8ckx4rAuA?pwd=kduw 提取码: kduw
# 预训练模型
采用pvt_v2_b2作为预训练模型进行训练，将其放入lib文件夹下即可，下载链接：通过网盘分享的文件：pvt_v2_b2.pth
链接: https://pan.baidu.com/s/1bqAies8SlG_hGT8b4WTbqg?pwd=ke2e 提取码: ke2e。也可采用其他基于transform架构的预训练模型进行训练，可以去官网下载对应的预训练模型
# 训练
相对应的环境以及预训练模型和数据集下载完毕，执行MyTrain.py即可进行模型的训练。
# 测试
对训练好的模型进行测试，所测试得到的并非是模型的性能评估指标，而是根据训练好模型得到的二值化图像。、
# 性能评估
可参考PySODMetrics-main项目，
