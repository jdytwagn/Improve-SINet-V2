# Improve-SINet-V2
基于深度学习的伪装目标检测网络模型设计及实现。
# 数据集
采用主流的四个数据集COD10K、CAMO、CHAMELEON 和 NC4K，数据集下载链接（百度网盘版）：通过网盘分享的文件：Dataset
链接: https://pan.baidu.com/s/159VrPMukt1Zii8ckx4rAuA?pwd=kduw 提取码: kduw
# 预训练模型
采用pvt_v2_b2作为预训练模型进行训练，将其放入lib文件夹下即可，下载链接：通过网盘分享的文件：pvt_v2_b2.pth
链接: https://pan.baidu.com/s/1bqAies8SlG_hGT8b4WTbqg?pwd=ke2e 提取码: ke2e。也可采用其他基于transform架构的预训练模型进行训练，可以去官网下载对应的预训练模型。
# 训练
相对应的环境以及预训练模型和数据集下载完毕，执行MyTrain.py即可进行模型的训练。已经训练好的模型链接：通过网盘分享的文件：KKK_best_47-5.7-ours.pth
链接: https://pan.baidu.com/s/1nz_E-tiGLIE8u1NN3rixJA?pwd=mgc1 提取码: mgc1。
# 测试
对训练好的模型进行测试，所测试得到的并非是模型的性能评估指标，而是根据训练好模型得到的二值化图像。
# 性能评估
可参考github中PySODMetrics-main项目，或根据新metric.py脚本进行模型的性能评估，在此脚本中需要上传原本的测试集中的GT图像，以及训练好的模型的道德测试图像。
# GUI界面
设计了一个简单的GUI界面，用户可以进行注册登录，或者修改主题背景，以及进行图片、视频或摄像头实时检测，也可随时停下保存。

![image](https://github.com/user-attachments/assets/0b463f55-029f-4fc6-80b2-c621b1f42aa5)

![image](https://github.com/user-attachments/assets/87cc27cf-98c6-4998-b531-85fa20b62ce1)




