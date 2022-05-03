#!/usr/bin/env python
# coding=utf-8
import os
import time
import torch
import math
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from pyg_DataSet import DataSet, StrongDataSet
from pyg_GCN import GCN, S_GCN
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

Epoch = 90  # 训练的次数
batch_size = 250  # 批大小
dataset = 'SIDER'


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \ log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

    # 加载数据
    data_set = DataSet(save_root=dataset + '/matrix', data_mode='train', dataset=dataset,
                       processed_file_names=dataset + '/trainData.pt')
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # data_set = StrongDataSet(save_root=dataset + '/Strong', data_mode='use', processed_file_names=dataset + '/trainStrong.pt')
    """
    data_set = torch.load(dataset + '/trainAddNoiseData.pt')
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(data_set['x'], data_set['y']),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=2)
    """
    # 定义模型
    # my_net = S_GCN()
    my_net = torch.load(dataset + '/128Compare0.pkl')
    # my_net = torch.load(dataset + '/crossloss13.pkl')
    # 迁移学习
    """
    for param in model_conv.parameters(): # 冻结权重
        param.requires_grad = False
    """
    num_ftrs = my_net.net.fc2_pyg.in_features
    my_net.net.fc2_pyg = nn.Linear(num_ftrs, 2)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    my_net = my_net.to(device)

    # 定义损失函数和优化器
    posi_rate = 0.5
    # criterion = FocalLoss(class_num=2, alpha=torch.tensor([1, 1]))
    criterion = FocalLoss(class_num=2, alpha=torch.tensor([posi_rate, 1 - posi_rate]), gamma=3)
    # criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params=my_net.parameters(), lr=0.001)

    optimizer_state = optimizer.state_dict()
    optimizer.load_state_dict(optimizer_state)

    # 训练模型
    my_net.train()

    num = 0
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()

        t = 0
        for data in train_loader:
            my_net.zero_grad()

            # predict_value = my_net(data[0])
            # loss = criterion(predict_value, data[1])
            predict_value, _ = my_net(data)
            loss = criterion(predict_value, data.y)
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()
            #lr_scheduler.step()

            t += 1

        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.20f} mins".format(epoch, 1000 * epoch_loss / t,
                                                                           (end_time - start_time) / 60))
        torch.save(my_net, dataset + '/3' + str(num) + '.pkl')
        num += 1
    # 保存模型
    # torch.save(my_net, dataset + '/test2.pkl')


if __name__ == '__main__':
    main()
