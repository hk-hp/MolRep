# coding=utf8
import os
import time
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from pyg_DataSet import DataSet, StrongDataSet
from pyg_GCN import CompareGCN
import torch.nn as nn
from torch.nn import init
from torch_geometric.nn import inits
from torch_geometric.data import Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Epoch = 16  # 训练的次数
batch_size = 300  # 批大小

dataset = 'SIDER'


class CompareLoss(nn.Module):
    def __init__(self, temperature=torch.tensor([0.1])):
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


# 生成图对比学习的模型
def CreateModel():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

    # 加载数据
    data_set = DataSet(save_root=dataset + '/matrix', data_mode='train', dataset=dataset,
                       processed_file_names=dataset + '/trainData.pt')
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # 定义模型
    # my_net = CompareGCN()

    my_net = torch.load('TOX21/0.2Compare 29.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    my_net = my_net.to(device)

    criterion = CompareLoss()
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.001)

    my_net.train()

    t = 0
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        num = 0
        for data in train_loader:
            my_net.zero_grad()

            commen_data, strong_data = my_net(data)
            loss = criterion(commen_data, strong_data, data.y)
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

            num += 1
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch,
                                                                          epoch_loss / num,
                                                                          (end_time - start_time) / 60))
        torch.save(my_net, dataset + '/Compare ' + str(t) + '.pkl')
        t += 1
    # 保存模型
    torch.save(my_net, dataset + '/Compare2.pkl')


# 生成图对比学习的数据
def CreateData(model: str):
    """
    :param model: 模型名
    :param mode:  生成数据集的类型(train,test,all)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

    # 加载数据
    mode = 'train'
    data_set = DataSet(save_root=dataset + '/matrix', data_mode=mode, dataset=dataset,
                       processed_file_names=dataset + '/' + mode + 'Data.pt')
    train_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)

    mode = 'test'
    data_set = DataSet(save_root=dataset + '/matrix', data_mode=mode, dataset=dataset,
                       processed_file_names=dataset + '/' + mode + 'Data.pt')
    test_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)

    my_net = torch.load(model)
    # 增强数据
    my_net.eval()
    strong_data_list = []

    true_super_data = torch.tensor([])
    false_super_data = torch.tensor([])

    with torch.no_grad():  # 关闭梯度
        for data in train_loader:
            strong_data, _ = my_net(data)
            if data.y.item() == 1:
                true_super_data = torch.cat([true_super_data, strong_data[0].view(1, -1)], dim=0)
            else:
                false_super_data = torch.cat([false_super_data, strong_data[0].view(1, -1)], dim=0)

    _, _, indices = choice(true_super_data, false_super_data)

    mode = 'train'
    with torch.no_grad():  # 关闭梯度
        for data in train_loader:
            strong_data, _ = my_net(data)
            # data.x[data.x.shape[0] - 1] = strong_data[0][indices]
            strong_data_list.append(Data(x=data.x,
                                         edge_index=data.edge_index,
                                         edge_attr=data.edge_attr,
                                         edge_inform=data.edge_inform,
                                         edge_num=data.edge_inform.shape[0],
                                         reversal=data.reversal[0],
                                         strong_feature=data.strong_feature,
                                         super=strong_data[0][indices].view(1, -1),
                                         y=data.y))

    StrongDataSet(save_root=dataset + '/' + mode + 'Strong', data_mode='create', strong_data=strong_data_list,
                  processed_file_names=dataset + '/' + mode + 'Strong.pt')

    strong_data_list = []
    mode = 'test'
    with torch.no_grad():  # 关闭梯度
        for data in test_loader:
            strong_data, _ = my_net(data)
            # data.x[data.x.shape[0] - 1] = strong_data[0][indices]
            strong_data_list.append(Data(x=data.x,
                                         edge_index=data.edge_index,
                                         edge_attr=data.edge_attr,
                                         edge_inform=data.edge_inform,
                                         edge_num=data.edge_inform.shape[0],
                                         reversal=data.reversal[0],
                                         strong_feature=data.strong_feature,
                                         super=strong_data[0][indices].view(1, -1),
                                         y=data.y))

    StrongDataSet(save_root=dataset + '/' + mode + 'Strong', data_mode='create', strong_data=strong_data_list,
                  processed_file_names=dataset + '/' + mode + 'Strong.pt')


# 筛选特征
def choice(true_data, false_data, num=128):
    para = (false_data.mean(0) - true_data.mean(0)).abs() / (false_data.std(0) + true_data.std(0))
    _, indices = para.sort(descending=True)
    indices = indices[0:num]
    false_data = false_data[:, indices]
    true_data = true_data[:, indices]
    return true_data, false_data, indices


# 生成热力图
def Create_pic(model: str, mode: str, sum):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

    # 加载数据
    data_set = DataSet(save_root=dataset + '/matrix', data_mode=mode, dataset=dataset,
                       processed_file_names=dataset + '/' + mode + 'Data.pt')
    test_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)

    my_net = torch.load(model)
    # 增强数据
    my_net.eval()
    true_data = torch.tensor([])
    false_data = torch.tensor([])
    with torch.no_grad():  # 关闭梯度
        for data in iter(test_loader):
            if true_data.shape[0] <= sum and data.y.item() == 1:
                strong_data, _ = my_net(data)
                true_data = torch.cat([true_data, strong_data[0].view(1, -1)], dim=0)
            elif false_data.shape[0] <= sum and data.y.item() == 0:
                strong_data, _ = my_net(data)
                false_data = torch.cat([false_data, strong_data[0].view(1, -1)], dim=0)
            elif false_data.shape[0] > sum and true_data.shape[0] > sum:
                break

    true_data, false_data, indices = choice(true_data, false_data)

    false_data = np.array(false_data)
    true_data = np.array(true_data)

    ax = sns.heatmap(false_data, annot=False, center=0)
    plt.show()
    ax = sns.heatmap(true_data, annot=False, center=0)
    plt.show()


if __name__ == '__main__':
    #dataset = 'BACE'
    #CreateData(dataset+'/128Compare30.pkl')
    #dataset = 'ClinTox'
    #(dataset+'/Compare2.pkl')
    #dataset = 'SIDER'
    #CreateData(dataset+'/Compare2.pkl')
    dataset = 'TOX21'
    #CreateData(dataset+'/Compare2.pkl')
    #dataset = 'HIV'
    #CreateData(dataset+'/128Compare3.pkl')
    #dataset = 'PCBA'
    #CreateModel()
    #dataset = 'HIV'
    #CreateModel()
    #dataset = 'MUV'
    CreateModel()
    #CreateData('128Compare30.pkl', 'train')
    # CheekData('Compare.pkl', 'train')
    #Create_pic('128Compare30.pkl', 'test', 10)
