import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch
import global_value
from pyg_Conv import SuperConv_N, SuperConv_E, glorot, zeros


class Block(nn.Module):
    def __init__(self, N_in_channels=128, N_out_channels=128, N_heads=4, E_in_channels=14, E_out_channels=14,
                 E_heads=4):
        super(Block, self).__init__()
        self.dropout = 0.5

        self.N_in_channels = N_in_channels
        self.N_out_channels = N_out_channels
        self.E_in_channels = E_in_channels
        self.E_out_channels = E_out_channels

        self.conv_N = SuperConv_N(in_channels=N_in_channels, out_channels=N_out_channels,
                                  edge_in_channels=E_out_channels, heads=N_heads)
        self.bn_N = pyg_nn.BatchNorm(N_out_channels)

        self.conv_E = SuperConv_E(in_channels=E_in_channels, out_channels=E_out_channels, heads=E_heads)
        self.bn_E = pyg_nn.BatchNorm(E_out_channels)

        #################################################################################################

        self.conv_N2 = SuperConv_N(in_channels=N_out_channels, out_channels=N_out_channels,
                                   edge_in_channels=E_out_channels, heads=N_heads)
        self.bn_N2 = pyg_nn.BatchNorm(N_out_channels)
        self.conv_E2 = SuperConv_E(in_channels=E_out_channels, out_channels=E_out_channels, heads=E_heads)
        self.bn_E2 = pyg_nn.BatchNorm(E_out_channels)

    def forward(self, x, edge_index, edge_inform, batch, BatchReversal, edge_num):
        out = self.conv_E(BatchReversal)
        out = self.bn_E(out)
        out = F.relu(out)

        if self.E_in_channels == self.E_out_channels:
            BatchReversal['feature'] = out + BatchReversal['feature']
        else:
            BatchReversal['feature'] = out
            """
            x_high = self.dim_E(BatchReversal['feature'].unsqueeze(2)).squeeze(2)
            BatchReversal['feature'] = out + x_high
            """

        edge_inform = BatchReversalToEdge(BatchReversal, batch)

        out = self.conv_N(x=x, edge_index=edge_index, edge_inform=edge_inform)
        out = self.bn_N(out)
        out = F.relu(out)

        if self.N_in_channels == self.N_out_channels:
            out = out + x
        """
        else:   
            x_high = self.dim_N(x.unsqueeze(2)).squeeze(2)
            out = out + x_high
        """
        #########################################################################
        out2 = self.conv_E2(BatchReversal)
        out2 = self.bn_E2(out2)
        out2 = F.relu(out2)
        BatchReversal['feature'] = -out2 + BatchReversal['feature']

        edge_inform = BatchReversalToEdge(BatchReversal, batch)

        out2 = self.conv_N2(x=out, edge_index=edge_index, edge_inform=edge_inform)
        out2 = self.bn_N2(out2)
        out2 = F.relu(out2)
        out2 = -out2 + out

        super_node = GetSuperNode(out2, batch)

        # out2 = F.dropout(out2, p=self.dropout, training=self.training)

        return out2, edge_inform, BatchReversal, super_node


# 获取关键结构,将非关键原子与插件超级节点的连接去掉
class GetImportantStruct(torch.nn.Module):
    def __init__(self, in_channels, k):
        """
        :param k: 关键结构阈值
        :param in_channels: 节点特征数
        """
        super(GetImportantStruct, self).__init__()
        self.k = k
        self.ComputeScore = nn.Parameter(torch.Tensor(1, in_channels))

    def reset_parameters(self):
        glorot(self.ComputeScore)

    def forward(self, x, edge_index, edge_num, edge_inform, batch, BatchReversal):
        edge_num_plus = torch.clone(edge_num)
        BatchReversal_plus = BatchReversal

        score = (x * self.ComputeScore).mean(-1).view(-1, 1)
        score = F.sigmoid(score)
        x_out = x * score * 2

        # 保存score
        # global_value.NUM += 1
        # torch.save(score, 'result/'+str(global_value.NUM)+'.pt')

        # 计算各个图节点数量
        num = 0
        # s = self.ComputeScore.grad
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
            graph_node_score.append(score[temp:temp + i, ])
            temp += i

        # 划分边为各个图边特征
        temp = 0
        edge = []
        for i in edge_num_plus:
            edge.append(edge_index[:, temp:temp + i])
            temp += i

        total_mask = torch.tensor([], dtype=torch.bool)
        super_edge = []

        before_edge_num = 0
        before_node_num = 0

        for _class, edge_x in enumerate(edge):
            super_edge_temp = []
            mask = torch.ones((edge_num_plus[_class],), dtype=torch.bool)
            new_edge_num = edge_num_plus[_class].item()

            for num in range(graph_node_score[_class].shape[0] - 1):
                s = graph_node_score[_class][num]
                if s < self.k:
                    super_edge_temp.append(num)
                    for i in range(edge_x.shape[1] - 1, 0, -1):
                        if edge_x[0][i].item() == num + before_node_num and \
                                edge_x[1][i].item() == graph_node_score[_class].shape[0] - 1 + before_node_num:
                            mask[i] = False
                            new_edge_num -= 1
                            break
                    # mask[edge_num[_class] - class_num[_class] + num + 1] = False
            super_edge.append(super_edge_temp)

            before_edge_num += edge_num_plus[_class]
            before_node_num += class_num[_class]

            total_mask = torch.cat((total_mask, mask), dim=0)
            edge_num_plus[_class] = new_edge_num

        edge_index_puls = edge_index[:, total_mask]
        edge_inform_plus = edge_inform[total_mask, :]
        #############################################################
        # 划分新边特征
        num = 0
        temp = 0
        class_num = []
        for i in BatchReversal_plus['batch']:
            if i.data == num:
                temp += 1
            else:
                num += 1
                class_num.append(temp)
                temp = 1
        class_num.append(temp)

        before_node_num = 0
        mask = torch.ones(BatchReversal_plus['feature'].shape[0], dtype=torch.bool)
        for _class, graph_super_edge in enumerate(super_edge):
            for i in graph_super_edge:
                i += BatchReversal_plus['com_num'][_class] + before_node_num
                mask[i] = False
            before_node_num += class_num[_class]

        leave_node = torch.arange(BatchReversal_plus['feature'].shape[0], dtype=torch.long)[mask]
        BatchReversal_plus['feature'], BatchReversal_plus['index'], BatchReversal_plus['batch'] = \
            filter_edge(BatchReversal_plus['feature'], BatchReversal_plus['index'], BatchReversal_plus['batch'],
                        leave_node)

        return x_out, edge_index_puls, edge_inform_plus, edge_num_plus, BatchReversal_plus


# 过滤被保留的边
def filter_edge(x, edge_index, batch, perm):
    """
    :param batch: 批信息
    :param x: 节点特征
    :param edge_index: 边索引
    :param perm: 被留下的边
    """
    num_nodes = x.shape[0]

    x = x[perm]
    batch = batch[perm]

    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    # edge_attr = edge_attr[mask]

    return x, torch.stack([row, col], dim=0), batch


# 反转图做成batch
def MakeBatchReversal(reversal, x, i):
    BatchReversal = {'feature': torch.tensor([]), 'index': torch.tensor([], dtype=torch.int64),
                     'batch': torch.tensor([], dtype=torch.int64), 'com_num': [], 'edge_num': []}
    node_num = 0
    for batch_num, ReversalData in enumerate(reversal):
        BatchReversal['feature'] = torch.cat((BatchReversal['feature'], ReversalData[0]), dim=0)
        BatchReversal['index'] = torch.cat((BatchReversal['index'], ReversalData[1] + node_num), dim=1)
        BatchReversal['batch'] = torch.cat((BatchReversal['batch'],
                                            torch.zeros(ReversalData[0].shape[0], dtype=torch.int64) + batch_num),
                                           dim=0)
        BatchReversal['com_num'].append(ReversalData[2])

        if len(ReversalData[1]) == 0:
            BatchReversal['edge_num'].append(0)
        else:
            BatchReversal['edge_num'].append(ReversalData[1].shape[1])

        node_num += ReversalData[0].shape[0]

    return BatchReversal


# 反转图转换到正常图
def BatchReversalToEdge(BatchReversal, batch):
    node_num = ComputeNodeNum(batch)
    # 划分新边特征
    num = 0
    temp = 0
    class_num = []
    for i in BatchReversal['batch']:
        if i.data == num:
            temp += 1
        else:
            num += 1
            class_num.append(temp)
            temp = 1
    class_num.append(temp)

    temp = 0
    sub_node = []
    for i in class_num:
        sub_node.append(BatchReversal['feature'][temp:temp + i, ])
        temp += i

    sub_edge_inform = torch.tensor([])
    temp = torch.zeros(1, BatchReversal['feature'].shape[1], dtype=torch.float32)
    temp[0][0] = 1.
    for i in range(len(sub_node)):
        for j in range(BatchReversal['com_num'][i]):
            sub_edge_inform = torch.cat((sub_edge_inform, sub_node[i][j].view(1, -1), sub_node[i][j].view(1, -1)),
                                        dim=0)
        # 自连边
        for j in range(node_num[i]):
            sub_edge_inform = torch.cat((sub_edge_inform, temp), dim=0)
        # 超级变
        sub_edge_inform = torch.cat((sub_edge_inform, sub_node[i][BatchReversal['com_num'][i]:]), dim=0)

    return sub_edge_inform


# 将子图整合成批
def make_SubGraph_batch(SubGraph: list):
    sub_x = torch.tensor([])
    sub_edge_index = torch.tensor([], dtype=torch.int64)
    sub_edge_attr = torch.tensor([])
    node_sum = 0
    for graph in SubGraph:
        sub_x = torch.cat((sub_x, graph['node_feature']), dim=0)
        sub_edge_index = torch.cat((sub_edge_index, node_sum + graph['matrix']), dim=1)
        sub_edge_attr = torch.cat((sub_edge_attr, graph['edge_feature']), dim=0)
        node_sum += graph['node_feature'].shape[0]
    return sub_x, sub_edge_index, sub_edge_attr


# 获取批中子图的超级节点
def Get_SubGraph_SuperNode(x: torch.tensor, SubGraph: list):
    """
    :param x: 所有节点特征
    :param SubGraph: 子图列表
    :return: 每张图的前k个超级节点
    """
    super_node = torch.zeros(len(SubGraph), x.shape[1])
    node_sum = 0
    for j, graph in enumerate(SubGraph):
        temp = torch.zeros(1, x.shape[1])

        SubGraph_sum = 0
        for i in graph['node_num']:
            temp = torch.cat((temp, x[i - 1 + SubGraph_sum + node_sum][:].unsqueeze(0)), dim=0)
            SubGraph_sum += i

        super_node[j][:] = temp.sum(0)
        node_sum += graph['node_feature'].shape[0]
    return super_node


# 获取批中的超级节点
def GetSuperNode(x: torch.tensor, batch: torch.tensor) -> torch.tensor:
    """
    :param x: 所有节点
    :param batch: 批索引
    :return: 每张图的超级节点
    """
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
    for i in class_num:
        graph.append(x[temp:temp + i, ])
        temp += i

    _, channel = x.size()  # channel:通道数
    result = torch.zeros([len(class_num), channel])

    for _class, graph_x in enumerate(graph):
        result[_class][:] = graph_x[len(graph_x) - 1][:]

    return result


# 计算各个图节点数量
def ComputeNodeNum(batch):
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
    return class_num
