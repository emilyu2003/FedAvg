import matplotlib.pyplot as plt

eps = ['0.5', '2', '8']


def readTXT(fileName, graphName, lineName):
    plt.figure()
    with open(fileName, 'r', encoding='utf-8') as f:
        k = 0
        for line in f.readlines():
            line = line.strip('\n')
            line = line.strip('[')
            line = line.strip(']')
            # print(line)
            tmp = line.split(',')
            for i in range(len(tmp)):
                tmp[i] = eval(tmp[i])
            print(tmp)
            plt.plot(range(len(tmp)), tmp, label=("eps = " + eps[k]))
            k += 1

    plt.ylabel(graphName)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    readTXT('./trainAccuracyRecord.txt', 'trainAccuracy', 'trainAcc')
    readTXT('./testAccuracyRecord.txt', 'testAccuracy', 'testAcc')
    readTXT('./trainLossRecord.txt', 'trainLoss', 'trainLoss')
    readTXT('./testLossRecord.txt', 'testLoss', 'testLoss')
