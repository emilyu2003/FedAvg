import csv
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn

from preprocess import GetDataSet


class Client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.train_losses = []

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        # 加载当前通信中最新的全局参数
        Net.load_state_dict(global_parameters, strict=True)
        # 加载本地数据
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        epochLoss = []
        epochAcc = []
        for epoch in range(localEpoch):
            batchLoss = []
            batchAcc = []
            for data, label in self.train_dl:
                # 加载到GPU上(如果有的话)
                data, label = data.to(self.dev), label.to(self.dev)
                # 初始化梯度
                opti.zero_grad()
                # 传入数据
                params = Net(data)
                # 计算损失函数
                loss = lossFun(params, label)
                # 反向传播
                loss.backward()
                # 裁剪梯度
                nn.utils.clip_grad_norm_(Net.parameters(), max_norm=4, norm_type=2)
                # 计算并更新梯度
                opti.step()
                batchLoss.append(loss.item())
                params = torch.argmax(params, dim=1)
                batchAcc.append((params == label).float().mean())
            epochLoss.append(sum(batchLoss) / len(batchLoss))
            epochAcc.append(sum(batchAcc) / len(batchAcc))
        # 返回当前Client基于自己的数据训练得到的新的模型参数和loss（用于作图）
        return Net.state_dict(), sum(epochLoss) / len(epochLoss), sum(epochAcc) / len(epochAcc)


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)

        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        # 划分数据
        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2

        # 将序列进行随机排序
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)

        for i in range(self.num_of_clients):
            # 两个数据切片
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            # 分配数据
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]

            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)

            # 创建一个客户端
            someone = Client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            # 为每个Client设置名字
            self.clients_set['client{}'.format(i)] = someone


if __name__ == "__main__":
    MyClients = ClientsGroup('mnist', True, 100, 0)
