#  针对已经存在网络模型 .pth 文件的情况
import netron

modelData = "./onnx/denoise.onnx"  # 定义模型数据保存的路径
netron.start(modelData)  # 输出网络结构