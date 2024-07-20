import torch
from torchvision import datasets,transforms
import torch.distributed.rpc as rpc
import os
from torch.utils.data import DataLoader, Dataset
from torch import nn
import pickle
import numpy as np

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
# 对数据集进行划分
def train_iid(dataset, workers):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_users = len(workers)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def test_iid(dataset, workers):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_users = len(workers)
    num_items = int(len(dataset)/num_users)

    # 这段代码主要是在dict_users字典中为每一个用户随机选择一组样本索引,
    # 并从all_idxs中移除已经选择的索引,以避免重复选择。
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    # 使用np.random.choice从all_idxs中随机选择num_items个索引,放入set中去重。
    # 将选择的索引存入dict_users,键为用户id,值为索引组成的set。
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        # 从all_idxs中移除已选择的索引,防止重复选择。
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# 获取数据
def getData(dataSet,workers):
    # apply_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataSet == "mnist":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('data/mnist', train=True, download=True,transform=transform_train)
        test_dataset = datasets.MNIST('data/mnist', train=False, download=True,transform=transform_test)
        
    else:
        transform_train=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # 注意加载训练集和测试集时使用的transform
        train_dataset = datasets.CIFAR10('data/cifar', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('data/cifar', train=False, download=True, transform=transform_test)
        # user_group = cifar_iid(train_dataset,workers)
        # print("cifar_iid data split compete")

    user_train_group = train_iid(train_dataset,workers)
    user_test_group = test_iid(test_dataset,workers)
    print("{} data split compete".format(dataSet))
    return train_dataset, test_dataset, user_train_group,user_test_group


# class DatasetSplit(Dataset):
#     """An abstract Dataset class wrapped around Pytorch Dataset class.
#     """

#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = [int(i) for i in idxs]

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return torch.tensor(image), torch.tensor(label)


# 各个客户端用来存储数据
class GetData(object):
    def __init__(self):
        # 注意两个数据集采用不同的路径 
        self.data_path = "data/worker_data/cifar10/" + rpc.get_worker_info().name
        # self.data_path = "data/worker_data/mnist/" + rpc.get_worker_info().name

    def save_train_data(self,dataset,idxs,worker_name):
        os.makedirs(self.data_path, exist_ok=True)
        data_file = os.path.join(self.data_path,f"{worker_name}_train_data.pkl")
        data = [(dataset[idx][0], dataset[idx][1]) for idx in idxs]
        with open(data_file,"wb") as f:
            pickle.dump(data,f)
        print(f"Worker {rpc.get_worker_info} 已将数据保存到 {data_file}")

    def save_test_data(self,dataset,idxs,worker_name):
        os.makedirs(self.data_path, exist_ok=True)
        data_file = os.path.join(self.data_path,f"{worker_name}_test_data.pkl")
        data = [(dataset[idx][0], dataset[idx][1]) for idx in idxs]
        with open(data_file,"wb") as f:
            pickle.dump(data,f)
        print(f"Worker {rpc.get_worker_info} 已将数据保存到 {data_file}")
    

if __name__ == '__main__':

    workers = ["worker01", "worker02", "worker03", "worker04"]

    # 获取数据
    print("Server getData start")
    train_dataset, test_dataset, user_train_groups, user_test_groups = getData("cifar10",workers)
    # train_dataset, test_dataset, user_train_groups, user_test_groups = getData("mnist",workers)
    print("Server getData complete")

    print("初始化rpc....")
    rpc.init_rpc("server", rank=0, world_size=5)
    
    # 向各个客户端发送数据
    for i in range(len(workers)):
        print("开始进行第{}个worker的数据传输".format(i+1))
        print("开始构建第{}个getData".format(i+1))
        getData = rpc.remote(workers[i],GetData)
        print("第{}个getData构建完成".format(i+1))

        print("开始进行第{}个worker的数据传输".format(i+1))
        getData.rpc_sync().save_train_data(train_dataset, user_train_groups[i],workers[i])
        getData.rpc_sync().save_test_data(test_dataset, user_test_groups[i],workers[i])
        print("第{}个worker的数据传输完成".format(i+1))

    print("--------------------向客户端发送数据完毕--------------------")

    rpc.shutdown()  