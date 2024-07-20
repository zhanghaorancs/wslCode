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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

global train_loader_iter
global worker_model
global worker_optimizer
global worker_criterion
global train_dataloader
global test_dataloader
global worker_scheduler
global outputs

class VGG16Client(nn.Module):
    def __init__(self, split_point):
        super(VGG16Client, self).__init__()
        self.split_point = split_point

        self.block1_part1 = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(3, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.block1_part2 = nn.Sequential(
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2_part1 = nn.Sequential(
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.block2_part2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3_part1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block3_part2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block3_part3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        if self.split_point == 1:
            x = self.block1_part1(x)
        elif self.split_point == 2:
            x = self.block1_part1(x)
            x = self.block1_part2(x)
        elif self.split_point == 3:
            x = self.block1_part1(x)
            x = self.block1_part2(x)
            x = self.block1_pool(x)
        elif self.split_point == 4:
            x = self.block1_part1(x)
            x = self.block1_part2(x)
            x = self.block1_pool(x)
            x = self.block2_part1(x)
        elif self.split_point == 5:
            x = self.block1_part1(x)
            x = self.block1_part2(x)
            x = self.block1_pool(x)
            x = self.block2_part1(x)
            x = self.block2_part2(x)
        elif self.split_point == 6:
            x = self.block1_part1(x)
            x = self.block1_part2(x)
            x = self.block1_pool(x)
            x = self.block2_part1(x)
            x = self.block2_part2(x)
            x = self.block2_pool(x)
        elif self.split_point == 7:
            x = self.block1_part1(x)
            x = self.block1_part2(x)
            x = self.block1_pool(x)
            x = self.block2_part1(x)
            x = self.block2_part2(x)
            x = self.block2_pool(x)
            x = self.block3_part1(x)
        return x

class VGG16Server(nn.Module):
    def __init__(self, split_point, num_classes=10):
        super(VGG16Server, self).__init__()
        self.split_point = split_point

        self.block1_part2 = nn.Sequential(
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2_part1 = nn.Sequential(
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.block2_part2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3_part1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block3_part2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block3_part3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            # nn.Linear(512 * 1 * 1, 4096),  # 输入为 512 * 1 * 1，因为 CIFAR-10 图像较小
            # nn.Linear(512 * 1 * 1, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.4),
            # # nn.Linear(4096, 4096),
            # nn.Linear(256, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.4),
            # # nn.Linear(4096, num_classes),
            # nn.Linear(256, num_classes),
            nn.Linear(512 * 1 * 1, 4096),  # 输入为 512 * 1 * 1，因为 CIFAR-10 图像较小
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Dropout(0.4),
            nn.Linear(4096, num_classes),

        )

    def forward(self, x):
        if self.split_point == 1:
            x = self.block1_part2(x)
            x = self.block1_pool(x)
            x = self.block2_part1(x)
            x = self.block2_part2(x)
            x = self.block2_pool(x)
            x = self.block3_part1(x)
            x = self.block3_part2(x)
            x = self.block3_part3(x)
            x = self.block3_pool(x)
            x = self.block4(x)
            x = self.block5(x)
        elif self.split_point == 2:
            x = self.block1_pool(x)
            x = self.block2_part1(x)
            x = self.block2_part2(x)
            x = self.block2_pool(x)
            x = self.block3_part1(x)
            x = self.block3_part2(x)
            x = self.block3_part3(x)
            x = self.block3_pool(x)
            x = self.block4(x)
            x = self.block5(x)
        elif self.split_point == 3:
            x = self.block2_part1(x)
            x = self.block2_part2(x)
            x = self.block2_pool(x)
            x = self.block3_part1(x)
            x = self.block3_part2(x)
            x = self.block3_part3(x)
            x = self.block3_pool(x)
            x = self.block4(x)
            x = self.block5(x)
        elif self.split_point == 4:
            x = self.block2_part2(x)
            x = self.block2_pool(x)
            x = self.block3_part1(x)
            x = self.block3_part2(x)
            x = self.block3_part3(x)
            x = self.block3_pool(x)
            x = self.block4(x)
            x = self.block5(x)
        elif self.split_point == 5:
            x = self.block2_pool(x)
            x = self.block3_part1(x)
            x = self.block3_part2(x)
            x = self.block3_part3(x)
            x = self.block3_pool(x)
            x = self.block4(x)
            x = self.block5(x)
        elif self.split_point == 6:
            x = self.block3_part1(x)
            x = self.block3_part2(x)
            x = self.block3_part3(x)
            x = self.block3_pool(x)
            x = self.block4(x)
            x = self.block5(x)
        elif self.split_point == 7:
            x = self.block3_part2(x)
            x = self.block3_part3(x)
            x = self.block3_pool(x)
            x = self.block4(x)
            x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
        # self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # if self.transform:
        #     if isinstance(image, torch.Tensor):
        #         image = image
        #     else:
        #         image = self.transform(image)
        return image, label


data_path = "data/worker_data/cifar10/worker03"
train_data_path = "worker03_train_data.pkl"
test_data_path = "worker03_test_data.pkl"

def weights_init(m):
    # if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight.data)
    # elif isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
def init_worker(data_path,train_data_path,test_data_path):
    global worker_model, worker_optimizer, worker_criterion, train_dataloader,test_dataloader, worker_scheduler,train_loader_iter,test_loader_iter
     # 加载数据
    # tarin_data_file = os.path.join("data/worker_datasplitClient01","splitClient01_train_data.pkl")
    # test_data_file = os.path.join("data/worker_datasplitClient01","splitClient01_test_data.pkl")
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
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    print("开始读取文件加载数据")
    with open(tarin_data_file, 'rb') as f:
        train_data = pickle.load(f)
        images, labels = zip(*train_data)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # train_dataset = CustomDataset(images, labels, transform_train)
    train_dataset = CustomDataset(images, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    train_loader_iter = iter(train_dataloader)
    
    # 打印train_dataset的大小
    print("train_dataset.shape:{}".format(len(train_dataset)))
    print("train_dataloader.shape:{}".format(len(train_dataloader)))

    with open(test_data_file, 'rb') as f:
        test_data = pickle.load(f)
        images, labels = zip(*test_data)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # test_dataset = CustomDataset(images, labels, transform_test)
    test_dataset = CustomDataset(images, labels)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader_iter = iter(test_dataloader)
    print("test_dataset.shape:{}".format(len(test_dataset)))
    print("test_dataloader.shape:{}".format(len(test_dataloader)))

    print("---------------------------------数据加载完成---------------------------------")
    # return train_loader_iter

    print("---------------------------------开始构建模型、优化器、损失函数---------------------------------")
    worker_model = VGG16Client(split_point=1)
    worker_model.apply(weights_init)
    # worker_optimizer = optim.SGD(worker_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    worker_optimizer = optim.Adam(worker_model.parameters(), lr=0.001)
    worker_criterion = nn.CrossEntropyLoss().to(device)
    # 学习率自动调整
    worker_scheduler = optim.lr_scheduler.StepLR(worker_optimizer, step_size=30, gamma=0.1)
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

    return outputs,targets

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

    return outputs,targets
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
        # gradients = torch.nn.functional.adaptive_avg_pool2d(gradients, outputs.shape[2:])
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


if __name__ == '__main__':
    print("---------------------------------worker1 RPC初始化---------------------------------")
    rpc.init_rpc("worker3",rank=3,world_size=5)

    rpc.shutdown()