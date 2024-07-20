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
# torch.autograd.set_detect_anomaly(True)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"


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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

# worker本地模型训练过程
# worker初始化
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
    # train_loader_iter = iter(train_dataloader)
    
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
    # test_loader_iter = iter(test_dataloader)
    print("test_dataset.shape:{}".format(len(test_dataset)))
    print("test_dataloader.shape:{}".format(len(test_dataloader)))

    print("---------------------------------数据加载完成---------------------------------")
    # return train_loader_iter

    print("---------------------------------开始构建模型、优化器、损失函数---------------------------------")
    worker_model = ResNet50Client()
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

    # 将梯度移动到当前设备（例如GPU）
    # gradients = [grad.to(device) for grad in gradients]

    gradients = gradients[0]
    if gradients.shape != outputs.shape:
        gradients = gradients.expand_as(outputs)
        # gradients = torch.nn.functional.adaptive_avg_pool2d(gradients, outputs.shape[2:])
    
    # gradients = gradients.to(outputs.device)
    # 使用返回的梯度执行本地模型的反向传播
    outputs = outputs.clone()
    outputs.backward(gradients,retain_graph=True)
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

workers = ["worker1","worker2","worker3","worker4"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 500
local_epoch = 30
selection_ratio = 1

if __name__ == '__main__':
    rpc.init_rpc("server",rank=0,world_size=5)
    # # 定义server模型、优化器、损失函数
    print("--------------------------定义server模型、优化器、损失函数--------------------------")
    server_model = ResNet50Server().to(device)
    server_model.apply(weights_init)
    # server_optimizer = optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9)
    server_optimizer = optim.Adam(server_model.parameters(), lr=0.001)
    server_criterion = nn.CrossEntropyLoss().to(device)
    server_scheduler = optim.lr_scheduler.StepLR(server_optimizer, step_size=30, gamma=0.1)

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
        # server_model.train()
        print("---------------------------开始第{}轮训练---------------------------".format(epoch+1))
        local_epoch_time = time.time()

        epoch_train_losses = []
        epoch_train_accuracies = []
        epoch_test_losses = []
        epoch_test_accuracies = []
        sequence_epoch_test_losses = []
        sequence_epoch_test_accuracies = []

        local_epoch_time = time.time()
        # 这里的local_epoch指的是worker的train_loader大小除以batch_size
        for e in range(local_epoch):
            e_time = time.time()

            global_train_outputs = []
            global_train_targets = [] 
            global_test_outputs = []
            global_test_targets = []

            # 每个小轮的训练轮次
            e_train_losses = []
            e_train_accuracies = []

            print("----------------------异步通知所有客户端开始本地模型训练----------------------")
            # 异步调用 worker_train_one_epoch 并等待结果
            train_futures = [rpc.rpc_async(worker, worker_train_one_epoch) for worker in workers]

            # 等待所有异步调用完成并获取结果
            train_results = [fut.wait() for fut in train_futures]
            
            train_server_model = [[] for _ in server_model.parameters()]
            for output,target in train_results:
                output = output.to(device)
                target = target.to(device)

                server_optimizer.zero_grad()
                preds = server_model(output)
                loss = server_criterion(preds, target)
                loss.backward()

                server_optimizer.step()

                # 计算Train accuracy
                _, predicted = torch.max(preds, 1)
                correct = (predicted == target).sum().item()
                accuracy = correct / target.size(0)

                # 记录损失和准确度
                e_train_losses.append(loss.item())
                e_train_accuracies.append(accuracy)

                # 存储各个客户端对应的模型梯度
                # for param_index, param in enumerate(server_model.parameters()):
                #     train_server_model[param_index].append(param.grad.clone())
                # train_server_model.append([param.grad.clone() for param in server_model.parameters()])
                # train_server_model.append(server_model)
            # print("----------train_server_model----------:{}".format(train_server_model))
            # 将结果添加到 global_outputs 和 global_targets
            # for output, target in train_results:
            #     global_train_outputs.append(output)
            #     global_train_targets.append(target)

            # print("----------------------开始拼接train outputs和targets----------------------")
            # # 此时需要拼接output和target
            # train_outputs = torch.cat(global_train_outputs,dim=0)
            # train_targets = torch.cat(global_train_targets,dim=0)
            # # outputs = outputs.view(-1, outputs.size(-1))  # 展平batch维度
            # # targets = targets.view(-1, targets.size(-1))  # 展平batch维度
            # train_outputs = train_outputs.to(device)
            # train_targets = train_targets.to(device)
            
            # server_optimizer.zero_grad()
            # preds = server_model(train_outputs)
            # loss = server_criterion(preds, train_targets)
            # loss.backward()

            e_avg_loss = sum(e_train_losses) / len(e_train_losses)
            e_avg_accuracy = sum(e_train_accuracies) / len(e_train_accuracies)
            e_train_losses.append(e_avg_loss)
            e_train_accuracies.append(e_avg_accuracy)

            # print("----------------------选择客户端参与梯度聚合----------------------")
            # # 选择客户端参与聚合
            # num_selected = int(len(workers) * selection_ratio)
            # if(selection_ratio == 1):
            #     selected_indices = [i for i in range(len(workers))]
            # else:
            #     # 采用随机选择方式
            #     selected_indices = random.sample(range(len(workers)), num_selected)
            #     selected_indices.sort()
                # 未被选择的客户端的索引
                # nonselected_indices = [i for i in range(len(workers)) if i not in selected_indices]

            # 这里选择变化最大的前50%梯度进行聚合
            # grad_magnitudes = [torch.norm(grad) for grad, _ in grad_client_map]
            # _, selected_indices = torch.topk(torch.tensor(grad_magnitudes), num_selected_grads)
            
            # 初始化用于保存选择的梯度
            # selected_grads = [[] for _ in server_model.parameters()]
            
            # 提取选中客户端的梯度
            # for idx in selected_indices:
            #     # print("train_server_model[0][idx]:{}".format(train_server_model[0][idx]))
            #     for param_index, param_grad in enumerate(train_server_model):
            #         selected_grads[param_index].append(param_grad[idx])
            #     # for param_index, param_grad in enumerate(train_server_model[idx]):
            #     #     selected_grads[param_index].append(param_grad.clone())
            # # print("----------selected_grads----------:{}".format(selected_grads))
            # # 执行梯度平均
            # avg_grads = [sum(grads) / len(grads) for grads in selected_grads]
            
            # 将平均后的梯度应用于模型
            # for param, avg_grad in zip(server_model.parameters(), avg_grads):
            #     param.grad = avg_grad

            # server_optimizer.step()

            # 计算Train accuracy
            # _, predicted = torch.max(preds, 1)
            # correct = (predicted == train_targets).sum().item()
            # accuracy = correct / train_targets.size(0)

            # # 记录损失和准确度
            # epoch_train_losses.append(loss.item())
            # epoch_train_accuracies.append(accuracy)

            # 获取服务器端模型的第一层梯度
            first_layer_grads = [param.grad.clone().cpu() for param in list(server_model.parameters())[:1]]

            # print("----------------------将聚合后的梯度返回给客户端----------------------")
            # # futures = [rpc.rpc_async(worker, worker_train_backward,args=first_layer_grads) for worker in workers]
            # futures = [rpc.rpc_async(workers[index], worker_train_backward,args=first_layer_grads) for index in nonselected_indices]
            # for idx, fut in enumerate(futures):
            #     fut.wait()

            # print("----------------------将未参与聚合的梯度返回给对应的客户端----------------------")
            # #将未被选择的参与聚合的梯度返回对应客户端执行本地模型反向传播
            # futures = [rpc.rpc_async(workers[idx], worker_train_backward, args=([train_server_model[0][idx].cpu()],)) for idx in nonselected_indices]
            # for idx, fut in enumerate(futures):
            #     fut.wait()
            e_end_time = time.time()
            print("------------------All worker_train_one_e_poch calls have completed.----------------")
        epoch_avg_loss = sum(e_train_losses) / len(e_train_losses)
        epoch_avg_acc = sum(e_train_accuracies) / len(e_train_accuracies)
        if(e % 5 == 0):
            print("第{}轮：第{}小轮：Train Loss: {}，Train Accuracy: {}".format((epoch+1),(e+1),epoch_avg_loss,epoch_avg_acc))
        with open("train_resnet50_cifar10_noAggregate_loss_accuracy.txt", "a") as f:
            f.write(f"{epoch+1}\t{e+1}\t{epoch_avg_loss}\t{epoch_avg_acc}\n")
            print("第{}轮：第{}小轮：用时：{}s".format((epoch+1),(e+1),(e_end_time-e_time)))
            # with open("train_e_time.txt", "a") as f:
            #     f.write(f"{epoch+1}\t{e+1}\t{e_end_time-e_time}\n")
        
        local_epoch_end_time = time.time()

        print("------------------异步调用客户端执行测试过程----------------")
            # 异步调用客户端使用test_dataloader进行测试进行测试
            # 异步调用客户端使用test_dataloader进行测试进行测试
        futures = [rpc.rpc_async(worker, worker_test_one_epoch) for worker in workers]
        # 等待所有异步调用完成并获取结果
        results = [fut.wait() for fut in futures]

        sequence_epoch_test_losses = []
        sequence_epoch_test_accuracies = []

        # server_model.eval()
        # 将结果添加到 global_outputs 和 global_targets
        for output, target in results:
            # global_test_outputs.append(output)
            # global_test_targets.append(target)

            output = output.to(device)
            target = target.to(device)

            preds = server_model(output)
            loss = server_criterion(preds, target)

            # 计算Test accuracy
            _, predicted = torch.max(preds, 1)
            correct = (predicted == target).sum().item()
            accuracy = correct / target.size(0)
            # 记录损失和准确度
            sequence_epoch_test_losses.append(loss.item())
            sequence_epoch_test_accuracies.append(accuracy)

            
            # print("----------------------开始拼接test outputs和targets----------------------")
            # 此时需要拼接output和target
            # test_outputs = torch.cat(global_test_outputs,dim=0)
            # test_targets = torch.cat(global_test_targets,dim=0)
            # # outputs = outputs.view(-1, outputs.size(-1))  # 展平batch维度
            # # targets = targets.view(-1, targets.size(-1))  # 展平batch维度
            # test_outputs = test_outputs.to(device)
            # test_targets = test_targets.to(device)

            # # server_optimizer.zero_grad()
            # preds = server_model(test_outputs)
            # loss = server_criterion(preds, test_targets)

            # # 计算Test accuracy
            # _, predicted = torch.max(preds, 1)
            # correct = (predicted == test_targets).sum().item()
            # accuracy = correct / test_targets.size(0)

            # epoch_test_losses.append(loss.item())
            # epoch_test_accuracies.append(accuracy)

        sequence_avg_loss = sum(sequence_epoch_test_losses) / len(sequence_epoch_test_losses)
        sequence_avg_accuracy = sum(sequence_epoch_test_accuracies) / len(sequence_epoch_test_accuracies)
        # # 记录损失和准确度
        # epoch_test_losses.append(sequence_avg_loss)
        # epoch_test_accuracies.append(sequence_avg_accuracy)
        # 打印损失
        print("第{}轮：Test Loss: {}，Test Accuracy: {}".format((epoch+1),sequence_avg_loss,sequence_avg_accuracy))
        # 更新学习率
        # server_scheduler.step()
        
        
        # 计算并输出平均损失和平均准确度
        # avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        # avg_train_accuracy = sum(epoch_train_accuracies) / len(epoch_train_accuracies)
        # avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
        # avg_test_accuracy = sum(epoch_test_accuracies) / len(epoch_test_accuracies)

        print("第{}轮：第{}小轮：用时：{}s".format((epoch+1),(e+1),(local_epoch_end_time-local_epoch_time)))
        
        with open("results/cifar10/test_resnet50_cifar10_noAggreagte_loss_acc.txt", "a") as f:
            f.write(f"{epoch+1}\t{e+1}\t{sequence_avg_loss}\t{sequence_avg_accuracy}\n")

    end = time.time()
    print(f"训练总时间: {end - epoch_start_time}s")

    rpc.shutdown()
