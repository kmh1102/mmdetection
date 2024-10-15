import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
# 创建一个简单的模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # 设置为评估模式

# 创建一个存储特征图的列表
feature_maps = []


# 定义钩子函数
def get_features(module, input, output):
    feature_maps.append(output)


def visualize_feature_maps(feature_maps):
    # 假设我们只看第一个特征图
    num_feature_maps = feature_maps[0].shape[1]  # 特征图的数量
    size = feature_maps[0].shape[2]  # 特征图的大小

    plt.figure(figsize=(20, 20))
    for i in range(num_feature_maps):
        plt.subplot(8, 8, i + 1)  # 假定最大 64 个特征图
        plt.imshow(feature_maps[0][0, i].cpu(), cmap='viridis')  # 绘制特征图
        plt.axis('off')
    plt.show()


# 注册钩子到特定层，这里选择 layer1 的第一个卷积块
hook = model.layer1[0].register_forward_hook(get_features)

if __name__ == '__main__':
    # 输入图像
    img_tensor = torch.randn(1, 3, 224, 224)  # 随机生成一个图像张量
    with torch.no_grad():  # 不需要计算梯度
        output = model(img_tensor)  # 前向传播

    # 现在 feature_maps 中包含了 layer1[0] 的输出特征图
    print(feature_maps[0].shape)  # 查看特征图的形状


    visualize_feature_maps(feature_maps)
    # 清理，取消注册钩子
    hook.remove()
