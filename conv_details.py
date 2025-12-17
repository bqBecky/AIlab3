import os, sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# set the current work direction
os.chdir(sys.path[0])
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# =======================================Part 1 加载数据 ================================================================
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# =======================================Part 1 定义网络模型 ============================================================
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(                               # 定义一个卷积层1，相继执行以下计算
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 灰度图片所以通道是1,16是输出通道，5是卷积核的大小为5X5的矩阵
            nn.ReLU(),                                             # 非线性变换
            nn.MaxPool2d(kernel_size=2, stride=2))                 # 最大池化操作，大小为2×2。其含义就是压缩图片，提高计算速度，挑选特征最明显的部分，去除其他多余部分
        self.layer2 = nn.Sequential(                                # 定义一个卷积层2，相继执行以下计算
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # 输入通道是16,32是输出通道，5是卷积核的大小为5X5的矩阵
            nn.ReLU(),                                              # 非线性变换
            nn.MaxPool2d(kernel_size=2, stride=2))                  # 最大池化操作，大小为2×2。其含义就是压缩图片，提高计算速度，挑选特征最明显的部分，去除其他多余部分
        self.fc = nn.Linear(7 * 7 * 32, num_classes)               # 全连接层，7×7×32为输入向量（经过2个卷积层后的图片）的维度，10为类别数

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# =======================================Part 3 训练网络模型 ============================================================
model = ConvNet(num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# =======================================Part 4 测试网络模型 ============================================================
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# 保存训练好的神经网络模型，供以后使用
torch.save(model.state_dict(), 'model.ckpt')