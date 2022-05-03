import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from pyg_Struct import Block, GetImportantStruct, MakeBatchReversal


from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat

NODE_NUM = 0

"""
class S_GCN(nn.Module):
    def __init__(self, is_compare_net=None):
        super(S_GCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 2)

        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)

        self.bn3 = nn.BatchNorm1d(50)

        self.pool = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.bn3.reset_parameters()

    def forward(self, data, is_compare_data=None):
        out = F.relu(self.conv1(data))
        out = self.bn1(out)
        out = self.pool(out)

        out = F.relu(self.conv2_drop(self.conv2(out)))
        out = self.bn2(out)
        out = self.pool(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.bn3(out)
        out = F.dropout(out, training=self.training)
        #out = F.dropout(out, p=self.dropout, training=self.training)

        out = F.log_softmax(self.fc2(out), dim=1)

        return out
"""
class S_GCN(nn.Module):
    def __init__(self, is_compare_net=None):
        super(S_GCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 100, 5)
        self.conv3 = nn.Conv2d(100, 300, 5)

        self.conv4 = nn.Conv2d(300, 300, 4)

        self.fc1 = nn.Linear(300, 50)
        self.fc2 = nn.Linear(50, 2)

        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(100)
        self.bn3 = nn.BatchNorm2d(300)
        self.bn4 = nn.BatchNorm2d(300)

        self.d_bn1 = nn.BatchNorm1d(50)

        self.pool = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv3.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.bn3.reset_parameters()
        self.bn4.reset_parameters()

        self.d_bn1.reset_parameters()

    def forward(self, data, is_compare_data=None):
        out = F.relu(self.conv1(data))
        out = self.bn1(out)
        out = self.pool(out)

        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.pool(out)

        out = F.relu(self.conv2_drop(self.conv3(out)))
        out = self.bn3(out)
        # out = self.pool(out)

        #out = F.relu(self.conv2_drop(self.conv4(out)))
        #out = self.bn4(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.d_bn1(out)
        out = F.dropout(out, training=self.training)

        out = F.log_softmax(self.fc2(out), dim=1)

        return out


class CompareGCN(nn.Module):
    def __init__(self):
        super(CompareGCN, self).__init__()

        self.net = GCN(is_compare_net=1)

    def forward(self, data):
        if self.training:
            out = self.net(data)
            # strong_out = self.net(data, is_compare_data=1)
            strong_out = None
        else:
            out = self.net(data)
            strong_out = None
        return out, strong_out


class GCN(nn.Module):
    def __init__(self, is_compare_net=None):
        super(GCN, self).__init__()
        # self.conv_E = torch.nn.Conv1d(in_channels=7)
        self.is_compare_net = is_compare_net

        self.dropout = 0.5

        self.Block1 = Block(N_in_channels=30, N_out_channels=128, E_in_channels=7, E_out_channels=14, )
        self.Block2 = Block()
        self.Block3 = Block()
        self.Block4 = Block()
        self.Block5 = Block()
        self.Block6 = Block()

        self.SelectStruct1 = GetImportantStruct(128, 0.5)
        self.SelectStruct2 = GetImportantStruct(128, 0.5)
        self.JL = pyg_nn.JumpingKnowledge('lstm', channels=128, num_layers=6)

        self.fc1_pyg = nn.Linear(128, 64)

        if is_compare_net is None:
            self.fc2_pyg = nn.Linear(64, 2)
        else:
            self.fc2_pyg = nn.Linear(64, 128)

    def forward(self, data, is_compare_data=None):
        if is_compare_data is None:
            x = data.x
        else:
            x = data.strong_feature

        edge_index = data.edge_index
        edge_inform = data.edge_inform
        batch = data.batch
        edge_num = data.edge_num
        reversal = data.reversal

        # 计算各个图节点数量
        num = 0
        temp = 0
        class_num = []
        for i in batch:
            if i.data == num:
                temp += 1
            else:
                num += 1
                class_num.append(temp)
                temp = 1
        class_num.append(temp)

        # 划分特征为各个图的特征
        temp = 0
        graph = []
        graph_node_score = []
        for i in class_num:
            graph.append(x[temp:temp + i, ])
            temp += i

        # 划分边为各个图边特征
        temp = 0
        edge = []
        for i in edge_num:
            edge.append(edge_index[:, temp:temp + i])
            temp += i

        BatchReversal = MakeBatchReversal(reversal, graph, edge)

        x_list = []
        # ############################## 图卷积原子特征部分 ############################## #
        out, edge_inform, BatchReversal, super_node = self.Block1(x, edge_index, edge_inform, batch, BatchReversal,
                                                                  edge_num)
        super_node = F.dropout(super_node, training=self.training, p=self.dropout)
        x_list.append(super_node)

        out, edge_inform, BatchReversal, super_node = self.Block2(out, edge_index, edge_inform, batch, BatchReversal,
                                                                  edge_num)
        super_node = F.dropout(super_node, training=self.training, p=self.dropout)
        x_list.append(super_node)

        out, edge_inform, BatchReversal, super_node = self.Block3(out, edge_index, edge_inform, batch, BatchReversal,
                                                                  edge_num)
        super_node = F.dropout(super_node, training=self.training, p=self.dropout)
        x_list.append(super_node)

        out, edge_index, edge_inform, edge_num, BatchReversal = self.SelectStruct2(out, edge_index, edge_num,
                                                                                   edge_inform,
                                                                                   batch, BatchReversal)

        out, edge_inform, BatchReversal, super_node = self.Block4(out, edge_index, edge_inform, batch, BatchReversal,
                                                                  edge_num)
        super_node = F.dropout(super_node, training=self.training, p=self.dropout)
        x_list.append(super_node)

        out, edge_inform, BatchReversal, super_node = self.Block5(out, edge_index, edge_inform, batch, BatchReversal,
                                                                  edge_num)
        super_node = F.dropout(super_node, training=self.training, p=self.dropout)
        x_list.append(super_node)

        out, edge_inform, BatchReversal, super_node = self.Block6(out, edge_index, edge_inform, batch, BatchReversal,
                                                                  edge_num)
        super_node = F.dropout(super_node, training=self.training, p=self.dropout)
        x_list.append(super_node)

        out = self.JL(x_list)

        out = F.relu(self.fc1_pyg(out))

        out = F.dropout(out, training=self.training, p=self.dropout)

        if self.is_compare_net is None:
            out = F.dropout(out, p=self.dropout, training=self.training)
            out = F.relu(self.fc2_pyg(out))
            out = F.log_softmax(out, dim=1)
        else:
            # out = F.tanh(self.fc2_pyg(out))
            out = F.log_softmax(self.fc2_pyg(out), dim=1)

        return out

    """ 
        # ############################## 子图特征部分 ############################## #

        x_sub, edge_index_sub, edge_attr_sub = make_SubGraph_batch(SubGraph)
        out_sub = F.relu(self.conv1_sub(x=x_sub, edge_index=edge_index_sub, edge_weight=edge_attr_sub))
        out_sub = self.bn1_sub(out_sub)
        x1_sub = Get_SubGraph_SuperNode(out_sub, SubGraph)

        out_sub = F.relu(self.conv2_sub(x=out_sub, edge_index=edge_index_sub, edge_weight=edge_attr_sub))
        out_sub = self.bn2_sub(out_sub)
        x2_sub = Get_SubGraph_SuperNode(out_sub, SubGraph)

        out_sub = self.SelectOut2(x1_sub, x2_sub)

        out_sub = F.relu(self.fc1_sub(out_sub))
        out_sub = F.sigmoid(self.fc2_sub(out_sub))


        out = F.sigmoid(out)

        # ############################## 分子特征部分 ############################## #
        out_M = F.relu(self.fc1_molecular(group_num))
        out_M = F.relu(self.fc2_molecular(out_M))
        out_M = F.sigmoid(out_M)

        # ############################## 特征结合部分 ############################## #
        out_sum = torch.cat((out, out_sub), 1)
        out_sum = F.relu(self.fc1_sum(out_sum))
    """


class GraphUNet(torch.nn.Module):
    """
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(GraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)
