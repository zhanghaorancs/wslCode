import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import torch.distributed.rpc as rpc
import pickle
from torchvision import datasets, transforms

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"


# 未用到
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

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
        #     image = self.transform(image)
        return image, label


class GetData(object):
    def __init__(self):
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

rpc.init_rpc("worker01",
                  rank=1,
                  world_size=5)

    
rpc.shutdown()    


# 为了测试数据的加载
# if __name__ == '__main__':

#     # 实际环境中将数据分发单独运行
#     # 加载数据
#     tarin_data_file = os.path.join("data/worker_datasplitClient01","splitClient01_train_data.pkl")
#     test_data_file = os.path.join("data/worker_datasplitClient01","splitClient01_test_data.pkl")
    
#     print("开始读取文件加载数据")
#     with open(tarin_data_file, 'rb') as f:
#         train_data = pickle.load(f)
#         images, labels = zip(*train_data)
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     train_dataset = CustomDataset(images, labels, transform)
#     train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    
#     # 打印train_dataset的大小

#     print("train_dataset.shape:{}".format(len(train_dataset)))
#     print("train_dataloader.shape:{}".format(len(train_dataloader)))


#     with open(test_data_file, 'rb') as f:
#         test_data = pickle.load(f)
#         images, labels = zip(*test_data)
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     test_dataset = CustomDataset(images, labels, transform)
#     test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4)
#     print("test_dataset.shape:{}".format(len(test_dataset)))
#     print("test_dataloader.shape:{}".format(len(test_dataloader)))

