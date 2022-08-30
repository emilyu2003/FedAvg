import copy
import os
import argparse

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
parser.add_argument('-ncomm', '--num_comm', type=int, default=50, help='number of communications')
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

    lossFunc = F.cross_entropy
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
    numInComm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 得到全局的参数
    globalParams = {}
    for key, var in net.state_dict().items():
        globalParams[key] = var.clone()

    trainLoss = []
    testLoss = []
    trainAcc = []
    testAcc = []
    # 通信
    for i in range(args['num_comm']):
        print("communication round {}".format(i + 1))

        # 对随机选的将100个客户端进行随机排序
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:numInComm]]

        sumParams = None
        lossLocals = []
        accLocals = []
        # 每个Client基于当前模型参数和自己的数据训练并更新模型
        for client in tqdm(clients_in_comm):
            # 获取当前Client训练得到的参数
            localParams, loss, acc = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'],
                                                                               net,
                                                                               lossFunc, opti, globalParams)
            # 对所有的Client返回的参数累加
            if sumParams is None:
                sumParams = {}
                for key, var in localParams.items():
                    sumParams[key] = var.clone()
            else:
                for var in sumParams:
                    sumParams[var] = sumParams[var] + localParams[var]
            lossLocals.append(copy.deepcopy(loss))
            accLocals.append(copy.deepcopy(acc))

        # 取平均值
        for var in globalParams:
            globalParams[var] = (sumParams[var] / numInComm)

        # 作图
        lossAvg = sum(lossLocals) / len(lossLocals)
        accAvg = sum(accLocals) / len(accLocals)
        print('Round {}, Average loss {}'.format(iter, lossAvg))
        trainLoss.append(lossAvg)
        trainAcc.append(float(accAvg))

        # 测试
        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(globalParams, strict=True)
                sumAcc = 0
                num = 0
                batch_loss = []
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    params = net(data)
                    loss = lossFunc(params, label)
                    params = torch.argmax(params, dim=1)
                    sumAcc += (params == label).float().mean()
                    num += 1
                    batch_loss.append(loss.item())
                print('accuracy: {}'.format(sumAcc / num))
                testAcc.append(float(sumAcc / num))
                testLoss.append(sum(batch_loss) / len(batch_loss))

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
    plt.plot(range(len(trainAcc)), trainAcc, label='trainAcc')
    plt.plot(range(len(testAcc)), testAcc, '--', label='testAcc')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(range(len(trainLoss)), trainLoss, label='trainLoss')
    plt.plot(range(len(testLoss)), testLoss, '--', label='testLoss')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    file = open(r".\trainAccuracyRecord.txt", "a")
    file.write(str(trainAcc))
    file.write("\n")
    file.close()

    file = open(r".\trainLossRecord.txt", "a")
    file.write(str(trainLoss))
    file.write("\n")
    file.close()

    file = open(r".\testAccuracyRecord.txt", "a")
    file.write(str(testAcc))
    file.write("\n")
    file.close()

    file = open(r".\testLossRecord.txt", "a")
    file.write(str(testLoss))
    file.write("\n")
    file.close()
