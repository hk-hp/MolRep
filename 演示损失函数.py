import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import time
import torch.optim as optim
import torch.nn.functional as F


Epoch = 120
batch_size = 100

mean1 = 4
str1 = 1
num1 = 500

mean2 = 3
str2 = 1
num2 = 100

class CompareLoss(nn.Module):
    def __init__(self, temperature=torch.tensor([0.5])):
        super(CompareLoss, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, inputs, strong_inputs, targets):
        posi = torch.tensor([])
        nega = torch.tensor([])

        strong_posi = torch.tensor([])
        strong_nega = torch.tensor([])
        # 分开正负样本
        for i in range(inputs.shape[0]):
            if targets[i].item() == 0:
                nega = torch.cat((nega, inputs[i].view(1, -1)), dim=0)
                strong_nega = torch.cat((strong_nega, strong_inputs[i].view(1, -1)), dim=0)

            else:
                posi = torch.cat((posi, inputs[i].view(1, -1)), dim=0)
                strong_posi = torch.cat((strong_posi, strong_inputs[i].view(1, -1)), dim=0)

        if len(posi) == 0:
            posi = inputs.mean(dim=0).unsqueeze(0)
            strong_posi = strong_inputs.mean(dim=0).unsqueeze(0)
        if len(nega) == 0:
            nega = inputs.mean(dim=0).unsqueeze(0)
            strong_nega = strong_inputs.mean(dim=0).unsqueeze(0)

        z = torch.cat((strong_posi, strong_nega, posi, nega), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, inputs.shape[0])
        sim_j_i = torch.diag(sim, -inputs.shape[0])

        # 正样本
        logits1 = torch.tensor([])
        for i in range(posi.shape[0]):
            positive_samples = sim_j_i[i].view(1)
            negative_samples = torch.cat((sim[i][posi.shape[0]:inputs.shape[0]],
                                          sim[i][posi.shape[0] + inputs.shape[0]:]), dim=0)

            r_positive_samples = sim_i_j[i].view(1)
            r_negative_samples = torch.cat((sim[i + inputs.shape[0]][posi.shape[0]:inputs.shape[0]],
                                            sim[i + inputs.shape[0]][posi.shape[0] + inputs.shape[0]:]), dim=0)

            logits1 = torch.cat(
                (logits1,
                 torch.cat((positive_samples, negative_samples), dim=0).unsqueeze(0),
                 torch.cat((r_positive_samples, r_negative_samples), dim=0).unsqueeze(0)),
                dim=0)

        labels = torch.zeros(logits1.shape[0]).long()
        loss1 = self.criterion(logits1, labels)

        # 负样本
        logits2 = torch.tensor([])
        for i in range(posi.shape[0], nega.shape[0] + posi.shape[0]):
            negative_samples = sim_j_i[i].view(1)
            positive_samples = torch.cat((sim[i][:posi.shape[0]],
                                          sim[i][inputs.shape[0]:posi.shape[0]]), dim=0)

            r_negative_samples = sim_i_j[i].view(1)
            r_positive_samples = torch.cat((sim[i + inputs.shape[0]][:posi.shape[0]],
                                            sim[i + inputs.shape[0]][inputs.shape[0]:posi.shape[0]]), dim=0)

            logits2 = torch.cat(
                (logits2,
                 torch.cat((negative_samples, positive_samples), dim=0).unsqueeze(0),
                 torch.cat((r_negative_samples, r_positive_samples), dim=0).unsqueeze(0)),
                dim=0)

        labels = torch.zeros(logits2.shape[0]).long()
        loss2 = self.criterion(logits2, labels)

        loss = (loss1 + loss2) / (targets.shape[0] * 2)
        return loss


class CompareLoss2(nn.Module):
    def __init__(self, temperature=torch.tensor([0.5])):
        super(CompareLoss2, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.mask = self.mask_correlated_samples(batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, inputs, strong_inputs, targets):
        N = 2 * inputs.shape[0]

        z = torch.cat((strong_inputs, inputs), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, inputs.shape[0])
        sim_j_i = torch.diag(sim, -inputs.shape[0])

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


def picturer1(x1, y1, x2, y2):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=9, xmin=0)
    plt.ylim(ymax=9, ymin=0)
    # 画两条（0-9）的坐标轴并设置轴标签x，y

    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4 ** 2  # 点面积
    # 画散点图
    plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
    plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='类别B')
    plt.legend()
    plt.show()


def picturer2(x1, y1, x2, y2):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=1, xmin=-1)
    plt.ylim(ymax=1, ymin=-1)

    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4 ** 2  # 点面积
    # 画散点图
    plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
    plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='类别B')
    plt.legend()
    plt.show()


class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()

        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)

        #self.fc3 = nn.Linear(16, 8)
        #self.fc4 = nn.Linear(8, 2)

        #self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out, negative_slope=0.01)
        #out = self.bn(out)
        out = self.fc2(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        return out


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.n = block()

    def forward(self, data):
        x = data[0]
        s_x = torch.clone(x) + (0.1 ** 0.5) * torch.randn(x.shape[0], x.shape[1])

        out = self.n(x)
        out1 = self.n(s_x)

        return out, out1


class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):  # 返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)


def demo(mode):
    x1 = np.random.normal(mean1, str1, num1)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
    y1 = np.random.normal(mean1, str1, num1)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
    x2 = np.random.normal(mean2, str2, num2)
    y2 = np.random.normal(mean2, str2, num2)
    picturer1(x1, y1, x2, y2)
    posi = torch.tensor(np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1))), dtype=torch.float32)
    nega = torch.tensor(np.hstack((x2.reshape(-1, 1), y2.reshape(-1, 1))), dtype=torch.float32)

    data = torch.cat((posi, nega), dim=0)
    labels = torch.cat((torch.ones(posi.shape[0]), torch.zeros(nega.shape[0])), dim=0)

    batch = torch.utils.data.DataLoader(
        MyDataset(data, labels), batch_size=batch_size, shuffle=True)

    # 定义模型
    my_net = net()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    my_net = my_net.to(device)
    if mode==0:
        criterion = CompareLoss()
    elif mode==1:
        criterion = CompareLoss2()
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.001)

    my_net.train()

    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        num = 0
        for data in batch:
            my_net.zero_grad()

            commen_data, strong_data = my_net(data)
            loss = criterion(commen_data, strong_data, data[1])
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

            num += 1
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch,
                                                                          epoch_loss / num,
                                                                          (end_time - start_time) / 60))
    # 保存模型
    if mode == 0:
        torch.save(my_net, 'demo0.pkl')
    elif mode==1:
        torch.save(my_net, 'demo1.pkl')


# 生成图对比学习的数据
def CreateData(model: str):
    """
    :param model: 模型名
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

    # 加载数据
    x1 = np.random.normal(mean1, str1, num1)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
    y1 = np.random.normal(mean1, str1, num1)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
    x2 = np.random.normal(mean2, str2, num2)
    y2 = np.random.normal(mean2, str2, num2)
    picturer1(x1, y1, x2, y2)

    posi = torch.tensor(np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1))), dtype=torch.float32)
    nega = torch.tensor(np.hstack((x2.reshape(-1, 1), y2.reshape(-1, 1))), dtype=torch.float32)

    data = torch.cat((posi, nega), dim=0)
    labels = torch.cat((torch.ones(posi.shape[0]), torch.zeros(nega.shape[0])), dim=0)

    batch = torch.utils.data.DataLoader(
        MyDataset(data, labels), batch_size=1, shuffle=True)

    my_net = torch.load(model)
    # 增强数据
    my_net.eval()

    s_p = torch.tensor([])
    s_n = torch.tensor([])
    with torch.no_grad():  # 关闭梯度
        for data in batch:
            commen_data, strong_data = my_net(data)
            if (data[1]) == 1:
                s_p = torch.cat((s_p, commen_data.view(1, -1)), dim=0)
            else:
                s_n = torch.cat((s_n, commen_data.view(1, -1)), dim=0)

    x1 = np.array(s_p)[:, 0]
    y1 = np.array(s_p)[:, 1]
    x2 = np.array(s_n)[:, 0]
    y2 = np.array(s_n)[:, 1]
    picturer2(x1, y1, x2, y2)


if __name__ == '__main__':
    demo(0)
    CreateData('demo0.pkl')
