import os, sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image


# set the current work direction
os.chdir(sys.path[0])
# Device configuration #如果有GPU的话就用GPU计算，否则就用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 指定相关参数
input_size = 784       # 输入图片的大小，28*28
hidden_size = 500      # 隐层神经元的个数（自己推定）
num_classes = 10       # 数字的类别0～9
num_epochs = 5         # 总循环的次数为5次，也就是在所有的图片上训练5趟。
batch_size = 100       # 每次批处理的图片的数量为100，每100个图片计算一次代价函数，更新一次参数
learning_rate = 0.001  # 学习的速率，即更新参数的步伐的大小
show_flag = True       # 是否显示输出信息


# =======================================Part 1 加载数据 ================================================================
# 加载 MNIST dataset 训练数据，总共60000张图片
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)

# 加载 MNIST dataset 测试数据，总共10000张图片
test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# 输出MNIST数据集的相关信息，显示训练集中的第1张图片data/img0.jpg
if show_flag:
    print('Training and test data:')
    print('    60000 Train images: ', train_dataset.data.shape)
    print('    10000 Test images: ', test_dataset.data.shape)
    # 显示训练集中的第1张图片
    img0 = Image.fromarray(train_dataset.data[0].numpy())
    img0.save('data/img0.jpg')


# 数据打包处理，也就是把每100（batch_size）个图片打成一个包，训练时一次提取一个包的图片
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,   #训练数据集
                                           batch_size=batch_size,   #包的大小，100张
                                           shuffle=True)            #加载的时候打乱图片顺序，消除结果的偶然性
# 跟训练集一样，打包处理
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)          #测试的时候不需要打乱图片顺序


# =======================================Part 1 定义网络模型 ============================================================
# 定义多层前馈神经网络（即定义一个类），包括一个输入层，一个隐藏层和一个输出层
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # 定义输入层(input_size)到隐藏层(hidden_size)的结构，为一个线性变换层(Linear)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 采用的非线性函数
        self.relu = nn.ReLU()
        # # 定义隐藏层(input_size)到输出层num_classes) 的结构，为另一个线性变换层(Linear)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # 显示两个线性层中的参数W1,b1,W2,b2的shape
        if show_flag:
            # parameters of Layer1 and Layer2
            print('Parameters:')
            print('    Layer 1 parameters shape: ', 'W1', self.fc1.weight.shape, 'b1', self.fc1.bias.shape)
            print('    Layer 2 parameters shape: ', 'W2', self.fc2.weight.shape, 'b2', self.fc2.bias.shape)
            print('    Layer 1 parameters data: ', 'W1', self.fc1.weight.data)
            # print('    Layer 1 parameters data: ', 'b1', self.fc1.bias.data)
            print('    Layer 2 parameters data: ', 'W2', self.fc2.weight.data)
            # print('    Layer 2 parameters data: ', 'b2', self.fc2.bias.data)

    # 多层前馈神经网络的前向计算过程，即输入图片，输出预测结果
    def forward(self, x):
        # 输入x是一个batch（100张）图片，即batch_size*input_size(100*768)
        if show_flag:
            print('Forward calculate to get prediction results:')
            print('    X, Input image shape (batch_size images): ', x.shape)

        # 经过第一层线性计算，即处理后的结果的shape为batch_size*hidden_size(100*500)
        out = self.fc1(x)
        if show_flag:
            print('    Layer 1 output shape: ', out.shape)

        # 非线性变换，shape保持不变
        out = self.relu(out)
        if show_flag:
            print('    ReLU output shape: ', out.shape)

        # 经过第二层线性计算，即处理后的结果的shape为batch_size*output_size(100*10)
        # 即这100经图片的预测结果，out的第一行即为第一张图片属于0－9这10个类别的“概率”
        # 这里其实并不是真正的“概率”，因为没有作softmax变换（用nn.CrossEntropyLoss计算代价时，softmax变换会自动加上）
        # 但out的第一行10个值中最大的那个就是第一张图片对应的类别
        out = self.fc2(out)
        if show_flag:
            print('    Layer 2 output shape: ', out.shape)

        # 返回预测的结果
        return out


# =======================================Part 3 训练网络模型 ============================================================
# 由类实例化对象，这里的model就是一个初始化了的多层前馈神经网络
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# 采用交叉熵代价函数
criterion = nn.CrossEntropyLoss()
# 定义一个优化器，更新模型的参数，学习率为learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练，优化参数
# 总共有多少个batch的训练数据，60000*100＝600个batch
total_step = len(train_loader)
# 在所有60000张训练图片上，训练5趟。这里的5趟是人为指定的，视情况而定。
for epoch in range(num_epochs):
    # 以每个batch（100张图片）为单位，预测分类结果，计算低价，更新参数
    for i, (images, labels) in enumerate(train_loader):
        # 一个batch的图片及其真实类别标记，二维图片拉成一维向量用作神经网络的输入
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        # 显示一个batch的图片的信息
        if show_flag:
            print('A batchsize (100) of images, update parameters per batchsize (100) images:')
            print('    A batchsize of images (one line in Tensor for a image): ', images.shape)
            print('    A batchsize of true labels for the 100 images):', labels.shape)

        # 调用model.forward()进行计算，得到预测的分类结果
        outputs = model(images)
        # 传入预测的分类结果和真实的类别标记，调用交叉熵代价函数计算loss
        loss = criterion(outputs, labels)
        # 梯度清零，
        optimizer.zero_grad()
        # 调用后向传播算法自动计算梯度（即每一个参数的偏导）
        loss.backward()
        
        # 对所有参数执行更新操作，例如参数 b = b - learning_rate * b的偏导数
        optimizer.step()

        # 显示偏导的相关信息
        if show_flag:
            print('Weights of Layer fc1 Gradients: ', model.fc1.weight.grad.shape, '\n')

        # 输入一些训练时的信息，如训练到多少轮，训练了多少个batch，当前的loss是多少
        show_flag = False
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# =======================================Part 4 测试网络模型 ============================================================
# 测试阶段,不需要调用反向传播算法计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    # 对所有测试的10000张图片，预测分类的结果，也是按batch进行（100张图片1个batch）
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        # 预测的分类结果，取最大值对应的index就是对应的类别
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # 比较预测的分类结果与真实的类别标记，统计预测对了多少张
        correct += (predicted == labels).sum().item()
    # 输出预测的准确率
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# 保存训练好的神经网络模型，供以后使用
torch.save(model.state_dict(), 'model.ckpt')
