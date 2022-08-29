import copy
import os
import argparse

from opacus import PrivacyEngine
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Model import MnistCNN, Mnist2NN
from Client import ClientsGroup
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端的数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=600, help='numer of the clients')
# 随机挑选的客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                    help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
# batchsize大小
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
# 模型名称
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
# 学习速率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset', "--dataset", type=str, default="mnist", help="需要训练的数据集")
# 模型验证频率（通信频率）
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=10, help='global model save frequency(of communication)')
# num_comm 通信次数
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    # 创建最后的结果
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    # 初始化模型
    if args['model_name'] == 'mnist_2nn':
        net = Mnist2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = MnistCNN()

    # 如果有多个GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    net = net.to(dev)

    loss_func = F.cross_entropy
    # 使用Adam下降法
    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    # 创建Clients群
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    # 差分隐私
    '''
    privacy_engine = PrivacyEngine()
    net, opti, testDataLoader = privacy_engine.make_private(
        module=net,
        optimizer=opti,
        data_loader=testDataLoader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )
    '''

    # 每次随机选取一定的Clients
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 得到全局的参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    loss_train = []
    accuracies = []
    # 通信
    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        # 对随机选的将100个客户端进行随机排序
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        loss_locals = []
        # 每个Client基于当前模型参数和自己的数据训练并更新模型
        for client in tqdm(clients_in_comm):
            # 获取当前Client训练得到的参数
            local_parameters, loss = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                               loss_func, opti, global_parameters)
            # 对所有的Client返回的参数累加
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
            loss_locals.append(copy.deepcopy(loss))

        # 取平均值
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        # 作图
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {}, Average loss {}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # 测试
        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    params = net(data)
                    params = torch.argmax(params, dim=1)
                    sum_accu += (params == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))
                accuracies.append(1 - sum_accu / num)

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
    # 绘图
    plt.figure()
    plt.plot(range(len(accuracies)), accuracies)
    plt.ylabel('test_accuracy')
    plt.show()
