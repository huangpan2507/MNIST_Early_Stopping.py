import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping


# Load and Batch the Data


def create_datasets(batch_size):

# percentage of training set to use as validation
    valid_size = 0.2


# convert data to torch.FloatTensor
    transform = transforms.ToTensor()

# choose the training and test dataset
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    num_train = len(train_data)              # 60000
    indices = list(range(num_train))
    print('indices:', indices, type(indices))
    np.random.shuffle(indices)
# np.floor向下取整
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samples for obtaining training and validation batches
    # 子集随机采样器，与shuffle互斥
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
    return train_loader, test_loader, valid_loader

# define the Network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.i = 0

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        self.i += 1
        # print('forward_i:', self.i)         # 188 for循环的次数
        return x


model = Net()
print('model:', model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# Train the Model using Early Stopping


def train_model(model, batch_size, patience, n_epoches):

    # to track the training loss as model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # 记录每个epoch中的valid_loss数据（每次epoch将loss放到数组里），用来画图显示出来
    avg_valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(1, n_epoches + 1):
        """ Train the model"""
        model.train()
        ii = 0
        for batch, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            ii += 1
        print('enumerate(train_loader)_i:', ii)
        print('enumerate(train_loader)_len:', len(train_loader))  # train_loader数据每组256，分了多少组
        train_losses.append(loss.item())
        print('train_losses', len(train_losses), train_losses)

# validation the model
        model.eval()
        j = 0
        # 只是用loss来评测模型参数好不好，不需要反向传播
        for data, target in valid_loader:
            # print('len_valid_loader:', len(data))       # 256
            j += 1
            output = model(data)
            loss = criterion(output, target)
        print('valid_loader_j:', j)              # valid_loader_j: 47
        valid_losses.append(loss.item())
        print('valid_losses', len(valid_losses), valid_losses)

    # calculate average loss over an epoch， 每个epoch中train_losses只有一个元素，average没作用
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epoches))                     # 将int转化为str， 再计算有几位数比如：n_epoches=100, 则有3位数

        print('epoch:', epoch, 'epoch_len:', epoch_len)
#  ^、<、>分别是居中、左对齐、右对齐，后面带宽度。  ：为格式限定符
        print_msg = (f'[{epoch: > {epoch_len}}/ {n_epoches: > {epoch_len}}]' + f'train_loss: {train_loss:.5f}' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

    # clear lists to track next epoch， 这里每个epoch后会清除，所以上面的np.average()没作用
        train_losses = []
        valid_losses = []

    # early_stopping needs the validation loss to check if it has decreased and if it has ,
    # it will make a checkpoint of the current model
    # 如果在创建class的时候写了call（）方法， 那么该class实例化出实例后， 实例名(xxx)就是调用call（）方法。
    # 此处调用early_stopping，会保存第一次及后面最好的状态，若状态变差，并连续几个epoch下滑，超过一定次数后将设置early_stop=True

        early_stopping(valid_loss, model)
        # 通过传入实例对象，和valid_loss来作为是否应该提前停止的依据，
        # 每个epoch都查看是否 应该提前停止，如果效果没之前好，则计数，再经过多少这种无效的训练就结束
        if early_stopping.early_stop:           # 不需要再训练，因为后面连续几个epoch均没有之前的好
            print('Early stopping')
            break

    # load the last checkpoint with the best model，经过训练epoch次数后，加载保存的状态
    model.load_state_dict(torch.load('checkpoint.pt'))
    print('model:', model, 'avg_train_losses:', avg_train_losses, 'avg_valid_losses:', avg_valid_losses)
    return model, avg_train_losses, avg_valid_losses


batch_size = 256
n_epoches = 5        # 100

train_loader, test_loader, valid_loader = create_datasets(batch_size)
patience = 3         # 20

model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epoches)


# visualize the loss as the network trained
# plt.figure()用来画图，自定义画布大小
fig = plt.figure(figsize=(10, 8))
# plt.plot(x ,y 坐标， 图示， 线条颜色， 线条样式， 标记样式)
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', color='r', linestyle='-', marker='1')
plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss', color='g', linestyle='--', marker='2')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss)) + 1
# 绘制一条横跨当前图表的垂直/水平辅助线
plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5)
plt.xlim(0, len(train_loss) + 1)
plt.grid(True)

# plt.legend()给图加上图示
plt.legend()

# tight_layout会自动调整子图参数，使之填充整个图像区域。
plt.tight_layout()
plt.show()
# bbox_inches可以剪出当前图表周围的空白部分,获得最小白边图像
fig.savefig('loss_plot.png', bbox_inches='tight')


# Test the Trained Network
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))   # list(0.0, 0.0, ....10个0.0)
print('class_correct:', class_correct)
class_total = list(0. for i in range(10))
print('class_total:', class_total)

model.eval()
k = 0
for data, target in test_loader:
    # data.size：[256, 1, 28, 28]  pred.shape：[256] len(data)：256  target.data ：[256]

    if len(target.data) != batch_size:
        break
    print('len_test_loader:', len(test_loader))    # 多少批次

    k += 1
    output = model(data)
    loss = criterion(output, target)

    # 比如test_loader=1000, batch_size=100, 则分成10组即循环10次， 每个的loss*循环次数（分组）*batch_size=total loss
    # print('k:', k)

    test_loss += loss.item() * data.size(0)
    _, pred = torch.max(output, 1)

    print('pred_eq_target:', pred.eq(target.data.view_as(pred)).size())  # torch.Size([256])

    # 通过np.squeeze()函数转换后，要显示的数组形状比如:[(1, 5) 此种情况无法显示]变成了秩为1的数组，即（5，）
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))   # torch.Size([256])

    print('correct_after_squeeze:', correct.shape, correct)    # torch.Size([256])

    for i in range(batch_size):
        # print('target_data:', target.data, 'i:', i)
        label = target.data[i]             # target.data数组中为0-9的数字
        print('label:', label)

        # 所有为i对应的label的预测正确次数之和，即可以知道数字为label的正确次数
        class_correct[label] += correct[i].item()
        # 一共有多少个 i对应的label
        class_total[label] += 1

test_loss = test_loss / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

# 统计每个label的正确率及所有准确率
for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%%(%2d/%2d)' % (str(i), 100 * class_correct[i] / class_total[i],
                                                        np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(
    class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))


# visualize Sample Test Result
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)

print('output.size:', output.size(), output)        # torch.Size([256, 10]

# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
# prep images for display，先转化为numpy类型
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    # 几行、几列、xticks 、yticks 为x，y的坐标轴刻度
    ax = fig.add_subplot(2, 20/2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')    # cmap输出指定颜色
    ax.set_title('{} ({})'.format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx] == labels[idx] else "red"))
fig.savefig('Pred_Target.png', bbox_inches='tight')
plt.show()



