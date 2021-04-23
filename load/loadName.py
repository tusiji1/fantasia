import torch.nn as tn
import torch.optim
import model.ecg_model as em
import torch as th
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import wfdb
import pywt
import numpy as np
import pandas
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_path = 'G:\\PostGraduate\\FANTASIA Nomal\\'
RATIO = 0.3

lr = 0.001
# 动态学习率
epochs = 30
tbs = 30
tebs = 10
success_rate = 0.0


# # 重采样
# def resampledata(data):



# 归一化
def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def getDataSet(data_path, number, i, data_set, lable_set):
    # print('读取'+number+'心电信号')
    record = wfdb.rdrecord(data_path + number, channel_names=['ECG'])
    data = record.p_signal[0:432000].flatten()
    # data = np.array(data)
    # data = minmaxscaler(data)
    # data = data.tolist()
    rdata = denoise(data)
    rdata1 = rdata[0:2000]
    plt.plot(rdata1)
    plt.show()
    data_set.append(rdata[0:432000])
    # should be 0
    # print(number[1:4])
    # number=int(number[1:4])
    x = np.array([i] * 864)
    list1 = x.tolist()
    lable_set.extend(list1)


def load(data_path):
    # should flect to 0-n
    number_set = ['f1o01', 'f1o02', 'f1o03', 'f1o04', 'f1o05', 'f1o06', 'f1o07', 'f1o08', 'f1o09', 'f1o10', 'f1y01',
                  'f1y02'
        , 'f1y03', 'f1y04', 'f1y05', 'f1y06', 'f1y07', 'f1y08', 'f1y09', 'f1y10']
    data_set = []
    lable_set = []
    i = 0
    for n in number_set:
        getDataSet(data_path, n, i, data_set, lable_set)
        i += 1
    # 转numpy数组,打乱顺序
    # print('读取完毕.........')
    data_set1 = np.array(data_set).reshape(-1, 500)
    lable_set1 = np.array(lable_set).reshape(-1, 1)
    train_ds = np.hstack((data_set1, lable_set1))

    np.random.shuffle(train_ds)
    X = train_ds[:, :500].reshape(-1, 500, 1)
    Y = train_ds[:, 500]
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


train_data = load(data_path)


class ecgDataSet(Dataset):
    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = train_data
        self.len = self.train_x.shape[0]

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.len


class ecgDataSetTest(Dataset):
    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = train_data
        self.len = self.test_x.shape[0]

    def __getitem__(self, index):
        return self.test_x[index], self.test_y[index]

    def __len__(self):
        return self.len


dataset = ecgDataSet()
datasetTest = ecgDataSetTest()

# 数据集 小批量 打乱 线程数

train_loader = DataLoader(dataset=dataset, batch_size=tbs, shuffle=True, num_workers=1)

test_loader = DataLoader(dataset=datasetTest, batch_size=tebs, shuffle=True, num_workers=1)

# if os.path.exists(project_path + 'success_rate_95.3lr_0.02epochs_17tbs_30tebs_10.pkl'):
#     model = th.load('success_rate_95.3lr_0.02epochs_17tbs_30tebs_10.pkl')
# else:
#     model = em.cnn_ecg_model()

model = em.cnn_ecg_model()


def train():
    criterion = tn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    for epoch in range(epochs):
        print('第%d轮学习.......' % (epoch + 1))
        for i, data in enumerate(train_loader, 0):
            inputs, target = data
            # inputs=th.unsqueeze(inputs,1)
            inputs = inputs.float()
            target = target.long()
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("损失率 %f" % loss)


def test():
    correct = 0
    total = 0
    print('开始测试.......')
    with th.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs = inputs.float()
            inputs = inputs.permute(0, 2, 1)
            target = target.long()
            outputs = model(inputs)
            _, predicted = th.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            rate = correct / total
    print('rate is %f' % rate)
    success_rate = rate
    model_name = 'success_rate_' + str(success_rate) + 'lr_' + str(lr) + 'epochs_' + str(epochs) + 'tbs_' + str(
        tbs) + 'tebs_' + str(tebs) + '.pkl'
    th.save(model, model_name)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # if os.path.exists(project_path + 'success_rate_95.3lr_0.02epochs_17tbs_30tebs_10.pkl'):
    #     print('存在模型开始测试')
    #     test()
    # else:
    #     print('开始训练')
    #     train()
    #     test()
    train()
    test()
