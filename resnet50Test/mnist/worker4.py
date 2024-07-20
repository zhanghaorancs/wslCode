import torch.distributed.rpc as rpc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
torch.autograd.set_detect_anomaly(True)


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

global train_loader_iter
global worker_model
global worker_optimizer
global worker_criterion
global train_dataloader
global test_dataloader
global worker_scheduler
global outputs

# 定义残差块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        # self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        out = self.conv2(out.clone())
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50Client(nn.Module):
    def __init__(self, split_point=1):
        super(ResNet50Client, self).__init__()
        self.split_point = split_point
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        # self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if self.split_point >= 2:
            self.layer1_1 = self._make_layer(Bottleneck, 64, 1)
        if self.split_point >= 3:
            self.layer1_2 = self._make_layer(Bottleneck, 64, 1)
        if self.split_point >= 4:
            self.layer1_3 = self._make_layer(Bottleneck, 64, 1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        if self.split_point >= 2:
            x = self.layer1_1(x)
        if self.split_point >= 3:
            x = self.layer1_2(x)
        if self.split_point >= 4:
            x = self.layer1_3(x)
        
        return x

class ResNet50Server(nn.Module):
    def __init__(self, split_point=1):
        super(ResNet50Server, self).__init__()
        self.split_point = split_point
        
        if self.split_point == 1:
            self.in_channels = 64
            self.layer1 = self._make_layer(Bottleneck, 64, 3)
            self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
            self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        elif self.split_point == 2:
            self.in_channels = 64 * Bottleneck.expansion  # 256
            self.layer1_2 = self._make_layer(Bottleneck, 64, 2)
            self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
            self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        elif self.split_point == 3:
            self.in_channels = 64 * Bottleneck.expansion  # 256
            self.layer1_3 = self._make_layer(Bottleneck, 64, 1)
            self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
            self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        elif self.split_point == 4:
            self.in_channels = 64 * Bottleneck.expansion  # 256
            self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
            self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.split_point == 1:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.split_point == 2:
            x = self.layer1_2(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.split_point == 3:
            x = self.layer1_3(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.split_point == 4:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 自定义的DataSet
class CustomDataset(Dataset):
    # def __init__(self, images, labels, transform=None):
    #     self.images = images
    #     self.labels = labels
    #     self.transform = transform
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


data_path = "data/worker_data/mnist/worker04"
train_data_path = "worker04_train_data.pkl"
test_data_path = "worker04_test_data.pkl"
def init_worker(data_path,train_data_path,test_data_path):
    global worker_model, worker_optimizer, worker_criterion, train_dataloader,test_dataloader, worker_scheduler,train_loader_iter,test_loader_iter
    # 加载数据
    tarin_data_file = os.path.join(data_path,train_data_path)
    test_data_file = os.path.join(data_path,test_data_path)
    
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_train = transforms.Compose(
    #     [transforms.Pad(4),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomGrayscale(),
    #     transforms.RandomCrop(32, padding=4),
        
    # ])

    # transform_test = transforms.Compose(
    #     [
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    # )
    
    print("开始读取文件加载数据")
    with open(tarin_data_file, 'rb') as f:
        train_data = pickle.load(f)
        images, labels = zip(*train_data)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
    # train_dataset = CustomDataset(images, labels, transform_train)
    train_dataset = CustomDataset(images, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_loader_iter = iter(train_dataloader)
    
    # 打印train_dataset的大小
    print("train_dataset.shape:{}".format(len(train_dataset)))
    print("train_dataloader.shape:{}".format(len(train_dataloader)))

    with open(test_data_file, 'rb') as f:
        test_data = pickle.load(f)
        images, labels = zip(*test_data)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
    # test_dataset = CustomDataset(images, labels, transform_test)
    test_dataset = CustomDataset(images, labels)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_loader_iter = iter(test_dataloader)
    print("test_dataset.shape:{}".format(len(test_dataset)))
    print("test_dataloader.shape:{}".format(len(test_dataloader)))

    print("---------------------------------数据加载完成---------------------------------")
    # return train_loader_iter

    print("---------------------------------开始构建模型、优化器、损失函数---------------------------------")
    # worker_model = ResNet50Client()
    worker_model = ResNet50Client(split_point=1)
    worker_model.apply(weights_init)
    worker_optimizer = optim.SGD(worker_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # worker_optimizer = optim.Adam(worker_model.parameters(), lr=0.001)
    worker_criterion = nn.CrossEntropyLoss()
    # 学习率自动调整
    # worker_scheduler = optim.lr_scheduler.StepLR(worker_optimizer, step_size=30, gamma=0.1)
    print("---------------------------------模型、优化器、损失函数构建完成---------------------------------")

# 执行本地模型前向传播
def worker_train_one_epoch():
    print("-------进入到{}的worker_train_one_epoch---------".format(rpc.get_worker_info().name))
    global worker_model, worker_optimizer, worker_criterion, train_dataloader, train_loader_iter,outputs
    worker_model.train()
    # try:
    #     inputs, targets = next(train_loader_iter)
    # except StopIteration:
    #     train_loader_iter = iter(train_dataloader)
    #     inputs, targets = next(train_loader_iter)

    # 从 DataLoader 中随机选择一个 batch
    batch_idx = random.randint(0, len(train_dataloader) - 1)
    for i, (inputs, targets) in enumerate(train_dataloader):
        if i == batch_idx:
            break
    
    worker_optimizer.zero_grad()
    outputs = worker_model(inputs)

    return outputs.cpu(), targets.cpu()

def worker_test_one_epoch():
    print("-------进入到{}的test_train_one_epoch---------".format(rpc.get_worker_info().name))
    global worker_model, worker_optimizer, worker_criterion, test_dataloader, test_loader_iter,outputs
    worker_model.eval()
    # try:
    #     inputs, targets = next(train_loader_iter)
    # except StopIteration:
    #     train_loader_iter = iter(train_dataloader)
    #     inputs, targets = next(train_loader_iter)
    
    # 从 DataLoader 中随机选择一个 batch
    batch_idx = random.randint(0, len(test_dataloader) - 1)
    for i, (inputs, targets) in enumerate(test_dataloader):
        if i == batch_idx:
            break
    
    # worker_optimizer.zero_grad()
    outputs = worker_model(inputs)

    return outputs.cpu(), targets.cpu()
    
def worker_train_backward(gradients):
    global worker_model, worker_optimizer, worker_criterion, train_dataloader, train_loader_iter,outputs
    # 调用 handle_output_fn 将结果发送给服务器，并等待服务器返回的梯度
    print("调用 worker_train_backward 执行客户端模型反向传播")
    # gradients_fut = rpc.rpc_sync("server", handle_worker_output, args=(outputs, targets))

    # 将梯度移动到当前设备（例如GPU）
    # gradients = [grad.to(device) for grad in gradients]

    gradients = gradients[0]
    if gradients.shape != outputs.shape:
        gradients = gradients.expand_as(outputs)
        # gradients = torch.nn.functional.adaptive_avg_pool2d(gradients, outputs.shape[2:]).squeeze(0)
    # gradients = gradients.to(outputs.device)
    outputs = outputs.clone()
    # 使用返回的梯度执行本地模型的反向传播
    # outputs.backward(gradients,retain_graph=True)
    outputs.backward(gradients)
    # outputs.backward(gradients)
    worker_optimizer.step()

def step_worker_scheduler():
    global worker_scheduler
    worker_scheduler.step()


def weights_init(m):
    # if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight.data)
    # elif isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)


if __name__ == '__main__':
    print("---------------------------------worker1 RPC初始化---------------------------------")
    rpc.init_rpc("worker4",rank=4,world_size=5)


    rpc.shutdown()