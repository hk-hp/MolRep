import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import OptPairTensor
from torch_geometric.utils import softmax
import math
import torch_geometric.nn as pyg_nn


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


# 针对节点的卷积层
class SuperConv_N(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_in_channels, heads=1):
        super(SuperConv_N, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.lin_l = Linear(in_channels + edge_in_channels, heads * out_channels, bias=False)  # 用于节点的注意力机制
        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))  # 用于节点的注意力机制
        self.bias = Parameter(torch.Tensor(out_channels))

        # self.node_feature_signal = Linear(in_channels, out_channels)  # 用于节点特征聚合后变换

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.node_feature_signal.weight)
        glorot(self.lin_l.weight)
        glorot(self.att_l)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_inform):
        x: OptPairTensor = (x, x)
        edge_inform: OptPairTensor = (edge_inform, edge_inform)

        # 计算节点的入度
        """
        degrees = torch.zeros(x[0].size(0))
        for i in edge_index[1][:]:
            degrees[i.item()] += 1
        """

        # 边分数计算
        x_l = torch.cat((x[0][edge_index[0]], edge_inform[0]), dim=1)
        x_l = self.lin_l(x_l).view(-1, self.heads, self.out_channels)

        alpha = (x_l * self.att_l).sum(dim=-1)
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index[1])
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        x_j = x_l * alpha.unsqueeze(-1)
        x_i = scatter(x_j, edge_index[1], dim=0, dim_size=edge_index[1].max() + 1, reduce='sum')

        out = x_i.mean(dim=1)  # 多头注意力求平均
        out += self.bias  # 加上偏置
        # 特征变化
        # x_i = self.node_feature_signal(x_i)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


# 针对边的卷积层
class SuperConv_E(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(SuperConv_E, self).__init__()
        self.conv = pyg_nn.GATConv(in_channels=in_channels, out_channels=out_channels, dropout=0, heads=heads, concat=False)

    def forward(self, BatchReversal):
        out = self.conv(x=BatchReversal['feature'], edge_index=BatchReversal['index'])

        return out


class weigaibianceluo(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_in_channels, edge_out_channels, heads=1):
        super(weigaibianceluo, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.lin_l = Linear(in_channels + edge_in_channels, heads * out_channels, bias=False)  # 用于节点的注意力机制
        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))  # 用于节点的注意力机制
        self.bias = Parameter(torch.Tensor(out_channels))

        self.edge_change = Linear(edge_in_channels, edge_out_channels)  # 用于边特征变化
        self.edge_attr = Linear(edge_in_channels, 1)  # 用于计算边权重

        # self.node_feature_signal = Linear(in_channels, out_channels)  # 用于节点特征聚合后变换

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.edge_attr.weight)
        glorot(self.edge_change.weight)
        # glorot(self.node_feature_signal.weight)
        glorot(self.lin_l.weight)
        glorot(self.att_l)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_inform, batch, edge_num):
        x: OptPairTensor = (x, x)
        edge_inform: OptPairTensor = (edge_inform, edge_inform)

        # 计算节点的入度
        degrees = torch.zeros(x[0].size(0))
        for i in edge_index[1][:]:
            degrees[i.item()] += 1

        # 边分数计算
        x_l = torch.cat((x[0][edge_index[0]], edge_inform[0]), dim=1)
        x_l = self.lin_l(x_l).view(-1, self.heads, self.out_channels)

        alpha = (x_l * self.att_l).sum(dim=-1)
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index[1])
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        x_j = x_l * alpha.unsqueeze(-1)
        x_i = scatter(x_j, edge_index[1], dim=0, dim_size=edge_index[1].max() + 1, reduce='sum')

        out = x_i.mean(dim=1)  # 多头注意力求平均
        out += self.bias  # 加上偏置
        # 特征变化
        # x_i = self.node_feature_signal(x_i)

        # 边特征聚合########################################
        # score = self.ComputeScore(x[0])

        # 计算各个图节点数
        num = 0
        temp = 0
        node_num = []
        for i in batch:
            if i.data == num:
                temp += 1
            else:
                num += 1
                node_num.append(temp)
                temp = 1
        node_num.append(temp)

        # 划分边为各个图边特征
        temp = 0
        sub_edge_index = []
        sub_edge_inform = []
        for i in edge_num:
            sub_edge_index.append(edge_index[:, temp:temp + i])
            sub_edge_inform.append(edge_inform[1][temp:temp + i, :])
            temp += i

        # 各个图边特征聚合
        super_node_num = 0
        gather_edge = torch.tensor([])
        for _class, a_edge_index in enumerate(sub_edge_index):
            edge_mask = torch.ones(a_edge_index.shape[1], dtype=torch.bool)

            # 去掉超级节点边和自连边
            for i in range(a_edge_index.shape[1]):
                if a_edge_index[1][i].item() == super_node_num + node_num[_class] - 1 or \
                        a_edge_index[1][i] == a_edge_index[0][i]:
                    edge_mask[i] = 0

            # 聚合普通边
            row_sub_edge_inform = sub_edge_inform[_class][edge_mask.T]
            row_sub_edge_index = sub_edge_index[_class][:, edge_mask]

            edge_i = scatter(row_sub_edge_inform, row_sub_edge_index[1] - super_node_num, dim=0, reduce='sum')

            for i in range(0, row_sub_edge_index.shape[1], 2):
                # 聚合相邻两个节点边特征
                temp = edge_i[row_sub_edge_index[0][i] - super_node_num] + edge_i[row_sub_edge_index[1][i] -
                                                                                  super_node_num] - row_sub_edge_inform[
                           i]
                # 计算边归一化参数
                edge_norm = degrees[row_sub_edge_index[0][i]] + degrees[row_sub_edge_index[1][i]] - 2

                temp /= (edge_norm - 1)
                gather_edge = torch.cat((gather_edge, temp.view(1, -1), temp.view(1, -1)), dim=0)

            # 聚合超级节点的有向边
            edge_i = scatter(sub_edge_inform[_class], sub_edge_index[_class][1] - super_node_num, dim=0, reduce='mean')

            gather_edge = torch.cat((gather_edge, edge_i, edge_i[0:edge_i.shape[0] - 1, :]), dim=0)  # 此处有bug，边和边信息不匹配
            super_node_num += node_num[_class]
        # ################################################

        # 边特征变化
        new_edge_inform = F.leaky_relu(self.edge_change(gather_edge))

        return out, new_edge_inform

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class weijiazhuyili(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_in_channels, edge_out_channels):
        super(weijiazhuyili, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.node_feature = Linear(in_channels, out_channels)
        self.node_feature_self = Linear(in_channels, out_channels)
        self.edge_change = Linear(edge_in_channels, edge_out_channels)
        self.edge_attr = Linear(edge_in_channels, 1)

        self.g = Linear(in_channels * 2, 1)
        self.node_feature_signal = Linear(in_channels, out_channels)

        self.ComputeScore = Linear(in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_feature.reset_parameters()
        self.node_feature_self.reset_parameters()
        self.edge_attr.reset_parameters()
        self.edge_change.reset_parameters()

        self.g.reset_parameters()
        self.node_feature_signal.reset_parameters()

    def forward(self, x, edge_index, edge_inform, batch, edge_num):
        x: OptPairTensor = (x, x)
        edge_inform: OptPairTensor = (edge_inform, edge_inform)

        # 计算节点的入度
        degrees = torch.zeros(x[0].size(0))
        for i in edge_index[1][:]:
            degrees[i.item()] += 1

        # 边分数计算
        edge_weight = F.leaky_relu(self.edge_attr(edge_inform[0]))  # 计算边权重

        # 节点聚合邻居边分数信息
        node_with_edge = scatter(edge_weight.view(-1, 1), edge_index[1], dim=0, reduce='mean')

        # 节点-边信息整合
        node_with_edge = node_with_edge * x[1]
        """
        # 计算门控参数
        h_ij = torch.cat((node_with_edge[edge_index[0]], node_with_edge[edge_index[1]]), dim=1)
        alphaG = F.tanh(self.g(h_ij))
        norm = (degrees[edge_index[0]] * degrees[edge_index[1]]).sqrt()
        alphaG = alphaG / norm

        # 特征聚合
        alphaG_j = alphaG.view(-1)[edge_index[0]]
        x_j = alphaG_j.view(-1, 1) * node_with_edge[edge_index[0]]
        x_i = scatter(x_j, edge_index[1], dim=0, reduce='sum')
        """
        x_j = node_with_edge[edge_index[0]]
        x_i = scatter(x_j, edge_index[1], dim=0, reduce='sum')

        # 特征变化
        x_i = self.node_feature_signal(x_i)

        # 边特征聚合########################################
        # score = self.ComputeScore(x[0])

        # 计算各个图节点数
        num = 0
        temp = 0
        node_num = []
        for i in batch:
            if i.data == num:
                temp += 1
            else:
                num += 1
                node_num.append(temp)
                temp = 1
        node_num.append(temp)
        """
        # 划分特征为各个图的特征
        temp = 0
        graph_edge_score = []
        for i in node_num:
            graph_edge_score.append(score[temp:temp + i, ])
            temp += i
        """
        # 划分边为各个图边特征
        temp = 0
        sub_edge_index = []
        sub_edge_inform = []
        for i in edge_num:
            sub_edge_index.append(edge_index[:, temp:temp + i])
            sub_edge_inform.append(edge_inform[1][temp:temp + i, :])
            temp += i

        # 各个图边特征聚合
        super_node_num = 0
        gather_edge = torch.tensor([])
        for _class, a_edge_index in enumerate(sub_edge_index):
            edge_mask = torch.ones(a_edge_index.shape[1], dtype=torch.bool)

            # 去掉超级节点边和自连边
            for i in range(a_edge_index.shape[1]):
                if a_edge_index[1][i].item() == super_node_num + node_num[_class] - 1 or \
                        a_edge_index[1][i] == a_edge_index[0][i]:
                    edge_mask[i] = 0

            # 聚合普通边
            row_sub_edge_inform = sub_edge_inform[_class][edge_mask.T]
            row_sub_edge_index = sub_edge_index[_class][:, edge_mask]
            """
            edge_score = F.softmax(graph_edge_score[_class][row_sub_edge_index[0] - super_node_num] +
                                   graph_edge_score[_class][row_sub_edge_index[1] - super_node_num], dim=0)

            row_sub_edge_inform = row_sub_edge_inform * edge_score
            """
            edge_i = scatter(row_sub_edge_inform, row_sub_edge_index[1] - super_node_num, dim=0, reduce='sum')

            for i in range(0, row_sub_edge_index.shape[1], 2):
                # 聚合相邻两个节点边特征
                temp = edge_i[row_sub_edge_index[0][i] - super_node_num] + edge_i[row_sub_edge_index[1][i] -
                                                                                  super_node_num] - row_sub_edge_inform[
                           i]
                # 计算边归一化参数
                edge_norm = degrees[row_sub_edge_index[0][i]] + degrees[row_sub_edge_index[1][i]] - 2

                temp /= (edge_norm - 1)
                gather_edge = torch.cat((gather_edge, temp.view(1, -1), temp.view(1, -1)), dim=0)

            # 聚合超级节点的有向边
            edge_i = scatter(sub_edge_inform[_class], sub_edge_index[_class][1] - super_node_num, dim=0, reduce='mean')

            gather_edge = torch.cat((gather_edge, edge_i, edge_i[0:edge_i.shape[0] - 1, :]), dim=0)  # 此处有bug，边和边信息不匹配
            super_node_num += node_num[_class]
        # ################################################

        # 边特征变化
        new_edge_inform = F.leaky_relu(self.edge_change(gather_edge))

        return x_i, new_edge_inform

    """
        # 计算门控参数
        h_ij = torch.cat((x[0][edge_index[0]], x[0][edge_index[1]]), dim=1)
        alphaG = F.tanh(self.g(h_ij))
        norm = (degrees[edge_index[0]] * degrees[edge_index[1]]).sqrt()
        alphaG = alphaG / norm

        # 特征聚合
        alphaG_j = alphaG.view(-1)[edge_index[0]]
        x_j = alphaG_j.view(-1, 1) * x[0][edge_index[0]]
        x_i = scatter(x_j, edge_index[1], dim=0, reduce='sum')

        # 特征变化
        x_i = self.node_feature_signal(x_i)

        # 边特征变化
        new_edge_inform = F.leaky_relu(self.edge_change(edge_inform[0]))

        return x_i, 
    """
    """
    def forward(self, x, edge_index, edge_inform, batch):
        x: OptPairTensor = (x, x)
        edge_inform: OptPairTensor = (edge_inform, edge_inform)

        # 边特征变化
        new_edge_inform = F.leaky_relu(self.edge_change(edge_inform[0]))

        # 边分数计算(AGG)
        edge_weight = F.leaky_relu(self.edge_attr(edge_inform[1]))  # 计算边权重
        edge_weight_j = edge_weight[edge_index[0]]
        # edge_weight_j = F.softmax(edge_weight_j, dim=0)
        x_j = edge_weight_j.view(-1, 1) * x[0][edge_index[0]]
        out_edge = scatter(x_j, edge_index[1], dim=0, reduce='sum')

        # 节点特征变化(TRANS)
        out = self.node_feature(out_edge)

        # 加入自身节点循环
        out_self = self.node_feature_self(x[1])

        return out + out_self, new_edge_inform
    """

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
