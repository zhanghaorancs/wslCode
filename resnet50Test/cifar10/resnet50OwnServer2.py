# 在测试过程中分别获得各个客户端测试集的损失和精度，然后取平均
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
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
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
        self.relu = nn.ReLU(inplace=True)
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
        # if self.transform:
        #     if isinstance(image, torch.Tensor):
        #         image = image
        #     else:
        #         image = self.transform(image)
        return image, label

# worker本地模型训练过程
# # worker初始化
def init_worker(data_path,train_data_path,test_data_path):
    global worker_model, worker_optimizer, worker_criterion, train_dataloader,test_dataloader, worker_scheduler,train_loader_iter,test_loader_iter
    tarin_data_file = os.path.join(data_path,train_data_path)
    test_data_file = os.path.join(data_path,test_data_path)
    
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    
    print("开始读取文件加载数据")
    with open(tarin_data_file, 'rb') as f:
        train_data = pickle.load(f)
        images, labels = zip(*train_data)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
    # train_dataset = CustomDataset(images, labels, transform)
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
    # test_dataset = CustomDataset(images, labels, transform)
    test_dataset = CustomDataset(images, labels)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_loader_iter = iter(test_dataloader)
    print("test_dataset.shape:{}".format(len(test_dataset)))
    print("test_dataloader.shape:{}".format(len(test_dataloader)))
    print("---------------------------------数据加载完成---------------------------------")
    print("---------------------------------开始构建模型、优化器、损失函数---------------------------------")
    worker_model = ResNet50Client(split_point=1)
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
    
    outputs = worker_model(inputs)

    return outputs.cpu(), targets.cpu()

def worker_train_backward(gradients):
    global worker_model, worker_optimizer, worker_criterion, train_dataloader, train_loader_iter,outputs
    # 调用 handle_output_fn 将结果发送给服务器，并等待服务器返回的梯度
    print("调用 worker_train_backward 执行客户端模型反向传播")

    gradients = gradients[0]
    if gradients.shape != outputs.shape:
        gradients = gradients.expand_as(outputs)

    # 使用返回的梯度执行本地模型的反向传播
    outputs.backward(gradients)

    worker_optimizer.step()

# 计算每个客户端梯度的变化量
def gradient_magnitude(gradient):
    return torch.norm(gradient).item()


workers = ["worker1","worker2","worker3","worker4"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 500
local_epoch = 30

# 初始化前一轮的梯度
previous_gradients = [None] * len(workers)

if __name__ == '__main__':
    rpc.init_rpc("server",rank=0,world_size=5)
    # # 定义server模型、优化器、损失函数
    print("--------------------------定义server模型、优化器、损失函数--------------------------")
    server_model = ResNet50Server(split_point=1).to(device)
    server_optimizer = optim.SGD(server_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # server_optimizer = optim.Adam(server_model.parameters(), lr=0.001)
    server_criterion = nn.CrossEntropyLoss().to(device)
    
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

    start = time.time()
    for epoch in range(epochs):
        print("---------------------------开始第{}轮训练---------------------------".format(epoch+1))
        local_epoch_time = time.time()
        # 这里的local_epoch指的是worker的train_loader大小除以batch_size
        for e in range(local_epoch):
            e_time = time.time()

            global_train_outputs = []
            global_train_targets = [] 
            global_test_outputs = []
            global_test_targets = [] 

            print("----------------------异步通知所有客户端开始本地模型训练----------------------")
            # 异步调用 worker_train_one_epoch 并等待结果
            futures = [rpc.rpc_async(worker, worker_train_one_epoch) for worker in workers]
            # 等待所有异步调用完成并获取结果
            results = [fut.wait() for fut in futures]

            # 记录每个客户端的输出大小
            output_sizes = []
            # 将结果添加到 global_outputs 和 global_targets
            for output, target in results:
                global_train_outputs.append(output)
                global_train_targets.append(target)
                output_sizes.append(output.size(0))

            print("----------------------开始拼接train outputs和targets----------------------")
            # 此时需要拼接output和target
            train_outputs = torch.cat(global_train_outputs,dim=0)
            train_targets = torch.cat(global_train_targets,dim=0)
            # outputs = outputs.view(-1, outputs.size(-1))  # 展平batch维度
            # targets = targets.view(-1, targets.size(-1))  # 展平batch维度
            train_outputs = train_outputs.to(device)
            train_targets = train_targets.to(device)

            server_optimizer.zero_grad()
            preds = server_model(train_outputs)

            e_losses = []
            e_acc = []
            # 分割output对应的pred到各个客户端，然后计算损失并执行反向传播，计算梯度
            start_idx = 0
            # worker_losses = []
            gradients_change = []
            for idx, worker in enumerate(workers):
                end_idx = start_idx + output_sizes[idx]
                worker_pred = preds[start_idx:end_idx]
                worker_target = train_targets[start_idx:end_idx]
                # worker_target = worker_target.to(worker_pred.device)

                worker_loss = server_criterion(worker_pred, worker_target)
                worker_loss.backward()
                gradients = [param.grad.clone() for param in server_model.parameters()]
                if previous_gradients[idx] is not None:
                    change = sum(gradient_magnitude(g - pg) for g, pg in zip(gradients, previous_gradients[idx]))
                else:
                    change = sum(gradient_magnitude(g) for g in gradients)
                gradients_change.append((worker, gradients, change))
                previous_gradients[idx] = gradients  # 更新前一轮的梯度

                server_optimizer.step()

                # worker_losses.append((worker, worker_loss))
                start_idx = end_idx

                e_losses.append(worker_loss.item())
                _,pre = torch.max(worker_pred.data, 1)
                acc = (pre == worker_target).sum().item() / worker_target.size(0)
                e_acc.append(acc)
            e_avg_loss = sum(e_losses) / len(e_losses)
            e_avg_acc = sum(e_acc) / len(e_acc)
            print("第{}轮：第{}小轮：Avg_loss:{},Avg_acc:{}".format((epoch+1),(e+1),e_avg_loss,e_avg_acc))
            # with open("results/cifar10/train_local_e_epoch_train_loss_acc.txt", "a") as f:
            #     f.write(f"{epoch+1}\t{e+1}\t{e_avg_loss}\t{e_avg_acc}\n")
            
            # loss = server_criterion(preds, train_targets)
            # loss.backward()

            # for name, param in server_model.named_parameters():
            #     print("name:{} ---- param.size:{}".format(name,param.size()))

            print("----------------------选择客户端参与梯度梯度变化较大的前50%梯度进行聚合----------------------")
            # gradients_change = []
            # for idx, (worker, worker_loss) in enumerate(worker_losses):
            #     worker_loss.backward(retain_graph=True)
            #     gradients = [param.grad.clone() for param in server_model.parameters()]
            #     if previous_gradients[idx] is not None:
            #         change = sum(gradient_magnitude(g - pg) for g, pg in zip(gradients, previous_gradients[idx]))
            #     else:
            #         change = sum(gradient_magnitude(g) for g in gradients)
            #     gradients_change.append((worker, gradients, change))
            #     previous_gradients[idx] = gradients  # 更新前一轮的梯度
            
            # 按变化量排序并选择变化较大的前50%
            gradients_change.sort(key=lambda x: x[2], reverse=True)
            selected_gradients = gradients_change[:len(gradients_change) // 2]

            # 聚合选择的梯度
            aggregated_gradient = [torch.zeros_like(param.grad) for param in server_model.parameters()]
            for _, gradients, _ in selected_gradients:
                for ag, g in zip(aggregated_gradient, gradients):
                    ag.add_(g)
            # aggregated_gradient = sum([gradient for _, gradient, _ in selected_gradients]) / len(selected_gradients)
            aggregated_gradient = [ag / len(selected_gradients) for ag in aggregated_gradient]
            
            # 将聚合后的梯度应用到模型参数上
            for param, gradient in zip(server_model.parameters(), aggregated_gradient):
                if param.grad is None:
                    param.grad = gradient.clone().detach()
                else:
                    param.grad.copy_(gradient)

            # # 计算Training accuracy
            # _, predicted = torch.max(preds, 1)
            # correct = (predicted == train_targets).sum().item()
            # accuracy = correct / train_targets.size(0)
            
            # 获取服务器端模型的第一层梯度
            first_layer_grads = [param.grad.clone().cpu() for param in list(server_model.parameters())[:1]]

            # 保存未被选择的客户端的第一层梯度
            # for idx, i in enumerate(nonselected_indices):
            #     nonselected_first_layer_grads[idx] = client_first_layer_grads[i]

            print("----------------------将聚合后的梯度返回给客户端----------------------")
            futures = [rpc.rpc_async(worker, worker_train_backward,args=first_layer_grads) for worker in workers]
            for idx, fut in enumerate(futures):
                fut.wait()

            # print("----------------------将未参与聚合的梯度返回给对应的客户端----------------------")
            # #将未被选择的参与聚合的梯度返回对应客户端执行本地模型反向传播
            # futures= [rpc.rps_async(workers[index], worker_train_backward,args=nonselected_first_layer_grads[index]) for index in nonselected_indices]
            # for idx, fut in enumerate(futures):
            #     fut.wait()

            print("------------------All worker_train_one_epoch calls have completed.----------------")
            print("第{}轮：第{}小轮：用时：{}s".format((epoch+1),(e+1),(time.time()-e_time)))

        print("------------------异步调用客户端执行测试过程----------------")
        # 异步调用客户端使用test_dataloader进行测试进行测试
        futures = [rpc.rpc_async(worker, worker_test_one_epoch) for worker in workers]

        # 等待所有异步调用完成并获取结果
        results = [fut.wait() for fut in futures]

        test_losses = []
        test_acces = []
        # 将结果添加到 global_outputs 和 global_targets
        for output, target in results:
            # global_test_outputs.append(output)
            # global_test_targets.append(target)
            output = output.to(device)
            target = target.to(device)

            preds = server_model(output)
            loss = server_criterion(preds,target)
            # 计算Test accuracy
            _, predicted = torch.max(preds, 1)
            correct = (predicted == target).sum().item()
            accuracy = correct / target.size(0)
            test_losses.append(loss.item())
            test_acces.append(accuracy)

        # print("----------------------开始拼接test outputs和targets----------------------")
        # # 此时需要拼接output和target
        # test_outputs = torch.cat(global_test_outputs,dim=0)
        # test_targets = torch.cat(global_test_targets,dim=0)
        # # outputs = outputs.view(-1, outputs.size(-1))  # 展平batch维度
        # # targets = targets.view(-1, targets.size(-1))  # 展平batch维度
        # test_outputs = test_outputs.to(device)
        # test_targets = test_targets.to(device)

        # server_optimizer.zero_grad()
        # preds = server_model(test_outputs)
        # loss = server_criterion(preds, test_targets)

        # # 计算Test accuracy
        # _, predicted = torch.max(preds, 1)
        # correct = (predicted == test_targets).sum().item()
        # accuracy = correct / test_targets.size(0)
        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_acc = sum(test_acces) / len(test_acces)

        # 打印损失
        print("第{}轮：第{}小轮：Test Loss: {}，Test Accuracy: {}".format((epoch+1),(e+1),avg_test_loss,avg_test_acc))

        print("第{}轮：第{}小轮：用时：{}s".format((epoch+1),(e+1),(time.time()-e_time)))
        with open("results/cifar10/test5010_sequence_sp1_acc.txt", "a") as f:
            f.write(f"{epoch+1}\t{e+1}\t{avg_test_loss}\t{avg_test_acc}\n")
        with open("results/cifar10/train5010_sequence_sp1_acc.txt", "a") as f:
                f.write(f"{epoch+1}\t{e+1}\t{e_avg_loss}\t{e_avg_acc}\n")

    end = time.time()
    print(f"训练总时间: {end - start}s")

    rpc.shutdown()
