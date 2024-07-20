import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import os
import torch.optim as optim
import time
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pickle
import copy
# torch.autograd.set_detect_anomaly(True)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# 多个客户端模型对应多个服务器模型
# SplitFed的实现

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
        # self.relu = nn.ReLU(inplace=False)
        self.relu = nn.ReLU()
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=False)
        self.relu = nn.ReLU()
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
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# worker本地模型训练过程
# worker初始化
def init_worker(data_path,train_data_path,test_data_path):
    global worker_model, worker_optimizer, worker_criterion, train_dataloader,test_dataloader, train_loader_iter,test_loader_iter,train_outputs,test_outputs
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
    # train_dataset = CustomDataset(images, labels, transform_train)
    train_dataset = CustomDataset(images, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 打印train_dataset的大小
    print("train_dataset.shape:{}".format(len(train_dataset)))
    print("train_dataloader.shape:{}".format(len(train_dataloader)))

    # with open(test_data_file, 'rb') as f:
    #     test_data = pickle.load(f)
    #     images, labels = zip(*test_data)
    # # test_dataset = CustomDataset(images, labels, transform_test)
    # test_dataset = CustomDataset(images, labels)
    # test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    test_dataset = datasets.CIFAR10('data/cifar', train=False, download=True, transform=transform_test)
    test_dataloader = DataLoader(test_dataset,batch_size=128,shuffle=False)
    print("test_dataset.shape:{}".format(len(test_dataset)))
    print("test_dataloader.shape:{}".format(len(test_dataloader)))

    print("---------------------------------数据加载完成---------------------------------")
    print("---------------------------------开始构建模型、优化器、损失函数---------------------------------")
    worker_model = ResNet50Client(split_point=1)
    worker_model.apply(weights_init)
    worker_optimizer = optim.SGD(worker_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # worker_optimizer = optim.Adam(worker_model.parameters(), lr=0.001)
    worker_criterion = nn.CrossEntropyLoss()
    # 学习率自动调整
    # worker_scheduler = optim.lr_scheduler.StepLR(worker_optimizer, step_size=30, gamma=0.1)
    print("---------------------------------模型、优化器、损失函数构建完成---------------------------------")

# 执行本地模型前向传播
def worker_train_one_epoch(batch_idx):
    print("-------进入到{}的worker_train_one_epoch---------".format(rpc.get_worker_info().name))
    global worker_model, worker_optimizer, worker_criterion, train_dataloader, train_loader_iter,train_outputs
    worker_model.train()
    if(batch_idx == 0):
        # 此时刚开始或者重新开始模型训练
        train_loader_iter = iter(train_dataloader)
        inputs, targets = next(train_loader_iter)
    else:
        inputs, targets = next(train_loader_iter)
    
    # 从 DataLoader 中随机选择一个 batch
    # batch_idx = random.randint(0, len(train_dataloader) - 1)
    # for i, (inputs, targets) in enumerate(train_dataloader):
    #     if i == batch_idx:
    #         break
    
    worker_optimizer.zero_grad()
    train_outputs = worker_model(inputs)
    train_outputs.retain_grad()  # 保留梯度
    train_outputs = train_outputs.clone()

    return train_outputs,targets

def worker_test_one_epoch(batch_idx):
    print("-------进入到{}的test_train_one_epoch---------".format(rpc.get_worker_info().name))
    global worker_model, worker_optimizer, worker_criterion, test_dataloader, test_loader_iter,test_outputs
    worker_model.eval()
    if(batch_idx == 0):
        # 此时刚开始或者重新开始模型训练
        test_loader_iter = iter(test_dataloader)
        inputs, targets = next(test_loader_iter)
    else:
        inputs, targets = next(test_loader_iter)
    
    # 从 DataLoader 中随机选择一个 batch
    # batch_idx = random.randint(0, len(test_dataloader) - 1)
    # for i, (inputs, targets) in enumerate(test_dataloader):
    #     if i == batch_idx:
    #         break
    
    test_outputs = worker_model(inputs)

    return test_outputs,targets

def worker_train_backward(gradients):
    global worker_model, worker_optimizer, worker_criterion, train_dataloader, train_loader_iter,train_outputs
    # 调用 handle_output_fn 将结果发送给服务器，并等待服务器返回的梯度
    print("调用 worker_train_backward 执行客户端模型反向传播")

    # gradients = gradients[0]
    # if gradients.shape != train_outputs.shape:
    #     gradients = gradients.expand_as(train_outputs)
    
    # 使用返回的梯度执行本地模型的反向传播
    train_outputs = train_outputs.clone()
    train_outputs.backward(gradients,retain_graph=True)
    # outputs.backward(gradients)
    worker_optimizer.step()
    return worker_model

def worker_train_backwardSFL(gradients):
    global worker_model, worker_optimizer, worker_criterion, train_dataloader, train_loader_iter,train_outputs
    # 调用 handle_output_fn 将结果发送给服务器，并等待服务器返回的梯度
    print("调用 worker_train_backward 执行客户端模型反向传播")

    gradients = gradients[0]
    if gradients.shape != train_outputs.shape:
        gradients = gradients.expand_as(train_outputs)
    
    # gradients = gradients.to(outputs.device)
    # 使用返回的梯度执行本地模型的反向传播
    train_outputs = train_outputs.clone()
    train_outputs.backward(gradients,retain_graph=True)
    # outputs.backward(gradients)
    worker_optimizer.step()

# 上传本地模型
def upLoadWorkerModel():
    global worker_model
    return worker_model.state_dict()

# 客户端更新本地model
def worker_model_update(recv_worker_model_state_dict):
    global worker_model
    worker_model.load_state_dict(recv_worker_model_state_dict)

def weights_init(m):
    # if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight.data)
    # elif isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)

# server中求Avg
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

workers = ["worker1","worker2","worker3","worker4"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 400
local_epoch = 196
selection_ratio = 1
test_epoch = 79

server_models = []
server_optimizers = []
server_criterions = []

if __name__ == '__main__':
    rpc.init_rpc("server",rank=0,world_size=5)
    # # 定义server模型、优化器、损失函数
    print("--------------------------定义server模型、优化器、损失函数--------------------------")
    # 为每一个worker都保留一个server_model
    for idx in range(len(workers)):
        server_models.append(ResNet50Server(split_point=1).to(device).apply(weights_init))
        server_optimizers.append(optim.SGD(server_models[idx].parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4))
        server_criterions.append(nn.CrossEntropyLoss().to(device))

    # server_model = ResNet50Server(split_point=1).to(device)
    # server_model.apply(weights_init)
    # server_optimizer = optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # server_optimizer = optim.Adam(server_model.parameters(), lr=0.001)
    # server_criterion = nn.CrossEntropyLoss().to(device)
    # server_scheduler = optim.lr_scheduler.StepLR(server_optimizer, step_size=30, gamma=0.1)

    # 通知worker初始化
    data_path = ["data/worker_data/cifar10/worker01","data/worker_data/cifar10/worker02",
                 "data/worker_data/cifar10/worker03","data/worker_data/cifar10/worker04"]
    train_data_path = ["worker01_train_data.pkl","worker02_train_data.pkl",
                       "worker03_train_data.pkl","worker04_train_data.pkl"]
    test_data_path = ["worker01_test_data.pkl","worker02_test_data.pkl",
                      "worker03_test_data.pkl","worker04_test_data.pkl"]

    print("--------------------------server端开始异步初始化workers--------------------------")
    futures = [rpc.rpc_async(workers[i],init_worker,args=(data_path[i],train_data_path[i],test_data_path[i])) for i in range(len(workers))]
    for fut in futures:
        fut.wait()
    
    print("--------------------------server端workers初始化完成--------------------------")
    print("--------------------------开始进行模型训练--------------------------")
    
    epoch_start_time = time.time()
    for epoch in range(epochs):
        print("---------------------------开始第{}轮训练---------------------------".format(epoch+1))
        epoch_train_losses = []
        epoch_train_accuracies = []

        local_epoch_time = time.time()
        # 这里的local_epoch指的是worker的train_loader大小除以batch_size
        for e in range(local_epoch):
            e_train_losses = []
            e_train_accuracies = []

            e_time = time.time()
            print("----------------------异步通知所有客户端开始本地模型训练----------------------")
            for idx in range(len(workers)):
                future = rpc.rpc_async(workers[idx],worker_train_one_epoch,args=(e,))
            # train_futures = [rpc.rpc_async(worker, worker_train_one_epoch) for worker in workers]
                worker_train_output, worker_train_target = future.wait()
                worker_train_output = worker_train_output.to(device)
                worker_train_target = worker_train_target.to(device)
                server_optimizers[idx].zero_grad()
                preds = server_models[idx](worker_train_output)
                loss = server_criterions[idx](preds,worker_train_target)
                loss.backward()
                server_optimizers[idx].step()

                # 计算Train accuracy
                _, predicted = torch.max(preds, 1)
                correct = (predicted == worker_train_target).sum().item()
                accuracy = correct / worker_train_target.size(0)

                # 记录损失和准确度
                e_train_losses.append(loss.item())
                e_train_accuracies.append(accuracy)

                # 获取server_model第一层的梯度
                first_layer_grads = [param.grad.clone().cpu() for param in list(server_models[idx].parameters())[:1]]
                future = rpc.rpc_async(workers[idx],worker_train_backwardSFL,args=first_layer_grads)
                # future = rpc.rpc_async(workers[idx],worker_train_backwardSFL,args=(worker_train_output.grad,))
                result = future.wait()

            e_avg_loss = sum(e_train_losses) / len(e_train_losses)
            e_avg_accuracy = sum(e_train_accuracies) / len(e_train_accuracies)
            print(f"{epoch+1}\t{e+1}\t{e_avg_loss}\t{e_avg_accuracy}\n")
            with open("results/cifar10/train_resnet5010_SFLNEW_sp1_loss_acc.txt", "a") as f:
                f.write(f"{epoch+1}\t{e+1}\t{e_avg_loss}\t{e_avg_accuracy}\n")

        print("----------------------客户端聚合----------------------")
        worker_model_weights = []
        server_model_weights = []
        for idx in range(len(workers)):
            # 获取各个客户端的本地模型
            future =rpc.rpc_async(workers[idx],upLoadWorkerModel)
            train_worker_model_statedict = future.wait()
            worker_model_weights.append(copy.deepcopy(train_worker_model_statedict))
            server_model_weights.append(copy.deepcopy(server_models[idx].state_dict()))
        # 执行本地模型聚合和服务器模型聚合
        Agg_worker_model_weight = average_weights(worker_model_weights)
        Agg_server_model_weight = average_weights(server_model_weights)

        # 更新各个客户端和服务器端模型
        for idx in range(len(workers)):
            server_models[idx].load_state_dict(Agg_server_model_weight)
            future = rpc.rpc_async(workers[idx],worker_model_update,args=(Agg_worker_model_weight,))
            result = future.wait()

        e_end_time = time.time()
        print("------------------All worker_train_one_epoch calls have completed.----------------")
        print("第{}轮：第{}小轮：用时：{}s".format((epoch+1),(e+1),(e_end_time-e_time)))
        # with open("train_e_time.txt", "a") as f:
        #     f.write(f"{epoch+1}\t{e+1}\t{e_end_time-e_time}\n")
        
        local_epoch_end_time = time.time()
        print("------------------执行测试过程----------------")
        # 应该直接调用workers[0]即可
        test_loss, test_total, test_correct = 0.0, 0.0, 0.0
        for test_e in range(test_epoch):
            future = rpc.rpc_async(workers[0],worker_test_one_epoch,args=(test_e,))
            worker_test_output,worker_test_target = future.wait()
            worker_test_output = worker_test_output.to(device)
            worker_test_target = worker_test_target.to(device)
            
            preds = server_models[0](worker_test_output)
            loss = server_criterions[0](preds, worker_test_target)
            test_loss += loss.item()

            # 计算Test accuracy
            _, predicted = torch.max(preds, 1)
            test_correct += (predicted == worker_test_target).sum().item()
            test_total += len(worker_test_target)

        test_avg_loss = test_loss / test_epoch
        test_avg_acc = test_correct / test_total

        print("第{}轮：第{}小轮：用时：{}s".format((epoch+1),(e+1),(local_epoch_end_time-local_epoch_time)))
        with open("results/cifar10/test_resnet5010_SFLNEW_sp1_loss_acc.txt", "a") as f:
            f.write(f"{epoch+1}\t{e+1}\t{test_avg_loss}\t{test_avg_acc}\n")

    end = time.time()
    print(f"训练总时间: {end - epoch_start_time}s")

    rpc.shutdown()
