import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import torch.distributed.rpc as rpc
from time import time
import pickle

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

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

rpc.init_rpc("worker04",
                  rank=4,
                  world_size=5)

rpc.shutdown()