import torch
from PIL import Image
import torchvision.transforms as transforms
# 从conv.py导入CNN模型类（确保conv.py中定义的模型类名为ConvNet）
from conv import ConvNet

# 配置设备（自动选择GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数（与训练时保持一致）
num_classes = 10  # 手写数字0-9共10类

def load_trained_model(model_path):
    """加载训练好的CNN模型"""
    # 初始化CNN模型
    model = ConvNet(num_classes).to(device)
    # 加载模型权重（已确认是CNN的权重）
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式（关闭 dropout、批归一化的训练模式）
    return model

def preprocess_image(image_path):
    """预处理单张图片：转为28x28灰度图，标准化（匹配MNIST数据集分布）"""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 缩放到MNIST的28x28尺寸
        transforms.Grayscale(),       # 转为单通道灰度图（MNIST是灰度图）
        transforms.ToTensor(),        # 转为Tensor格式（值范围0-1）
        transforms.Normalize(         # 用MNIST数据集的均值和标准差标准化
            mean=(0.1307,),           # MNIST训练集均值
            std=(0.3081,)             # MNIST训练集标准差
        )
    ])
    # 打开图片并应用预处理
    image = Image.open(image_path)
    # 增加批次维度（模型要求输入格式为[batch_size, channels, height, width]）
    return transform(image).unsqueeze(0).to(device)

if __name__ == "__main__":
    # 待预测的图片路径（你的img0.jpg）
    image_path = "data/img0.jpg"
    # CNN模型权重路径（你的model.ckpt）
    model_path = "model.ckpt"

    try:
        # 加载模型和图片
        model = load_trained_model(model_path)
        image = preprocess_image(image_path)

        # 无梯度预测（节省计算资源）
        with torch.no_grad():
            output = model(image)  # 模型输出（10个类别的概率分布）
            _, predicted = torch.max(output.data, 1)  # 取概率最大的类别

        # 输出预测结果
        print(f"图片 {image_path} 的预测结果为：{predicted.item()}")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e.filename}")
    except Exception as e:
        print(f"预测过程出错：{str(e)}")