import numpy as np
from torchvision import transforms  # 对图像进行原始的数据处理的工具
from torchvision import datasets  # 获取数据
from torch.utils.data import DataLoader  # 加载数据


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        # 训练集
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.train_loader = None
        # 测试数据集
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self.test_loader = None

        self._index_in_train_epoch = 0

        self.mnistDataSetConstruct(isIID)

    # mnistDataSetConstruct 数据重构
    def mnistDataSetConstruct(self, isIID):
        # nonIID:按数字大小排序，划分为200组大小为300的数据切片，2个切片/Client
        # isIID:随机600个样本/Client

        # 加载数据集
        batch_size = 100  # 设置训练的batch为10
        # 把训练数据转换成torch中的tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root='./dataset',
                                       train=True,
                                       download=False,
                                       transform=transform)

        test_dataset = datasets.MNIST(root='./dataset',
                                      train=False,
                                      download=False,
                                      transform=transform)

        self.train_data_size = len(train_dataset)
        self.test_data_size = len(test_dataset)

        train_data_np = train_dataset.data.numpy()
        train_label_np = train_dataset.targets.numpy()
        train_label_np = oneHot(train_label_np)

        test_data_np = test_dataset.data.numpy()
        test_label_np = test_dataset.targets.numpy()
        test_label_np = oneHot(test_label_np)

        train_data_np = train_data_np.reshape(train_data_np.shape[0], train_data_np.shape[1] * train_data_np.shape[2])
        test_data_np = test_data_np.reshape(test_data_np.shape[0], test_data_np.shape[1] * test_data_np.shape[2])

        # 进行归一化处理(数组每个位置都乘上1/255)
        train_data_np = train_data_np.astype(np.float32)
        train_data_np = np.multiply(train_data_np, 1.0 / 255.0)
        test_data_np = test_data_np.astype(np.float32)
        test_data_np = np.multiply(test_data_np, 1.0 / 255.0)

        # 添加噪声
        sensitivity = 1
        eps = 0.5
        train_data_np = laplaceMech(train_data_np, sensitivity, eps)
        self.test_data = test_data_np
        self.test_label = test_label_np

        if isIID:  # 随机
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_data_np[order]
            self.train_label = train_label_np[order]
        else:  # 按标签排序
            labels = np.argmax(train_label_np, axis=1)
            order = np.argsort(labels)
            self.train_data = train_data_np[order]
            self.train_label = train_label_np[order]

        self.train_loader = DataLoader(self.train_data,
                                       shuffle=False,
                                       batch_size=batch_size)
        self.test_loader = DataLoader(self.test_data,
                                      shuffle=False,
                                      batch_size=batch_size)

        '''print(self.train_data.shape)
        print(self.test_data.shape)
        print(self.train_label.shape)
        print(self.test_label.shape)'''


def oneHot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def noisyCount(sensitivety, epsilon):
    beta = sensitivety / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    return n_value


def laplaceMech(data, sensitivity, epsilon):
    for i in range(len(data)):
        data[i] += noisyCount(sensitivity, epsilon)
    return data


if __name__ == "__main__":
    'test data set'
    mnistDataSet = GetDataSet('mnist', 0)  # 0:NON-IID 1:IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    # print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    # print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    # print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
