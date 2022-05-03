from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolDescriptors, EState, FragmentCatalog, BRICS
import torch
import numpy as np
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import random
import math
import global_value
import deepchem as dc

is_pass = 0  # 0代标不处理


# 特征归一化
def normalization(feature, row_begin_end=None):
    """
    :param feature: 需要归一化的数据
    :param row_begin_end: 归一化数据开始的行(beg, end)，包括beg, 不包括end
    """
    if row_begin_end is None:
        rows, cols = feature.shape
        for i in range(cols):
            if feature[:, i].std() == 0:
                new_col = feature[:, i] - feature[:, i].mean()
            else:
                new_col = (feature[:, i] - feature[:, i].mean()) / feature[:, i].std()
            feature[:, i] = new_col

    else:
        norm_feature = feature[row_begin_end[0]: row_begin_end[1], :]
        rows, cols = norm_feature.shape
        for i in range(cols):
            if norm_feature[:, i].std() == 0:
                new_col = norm_feature[:, i] - norm_feature[:, i].mean()
            else:
                new_col = (norm_feature[:, i] - norm_feature[:, i].mean()) / norm_feature[:, i].std()
            norm_feature[:, i] = new_col

        feature[row_begin_end[0]: row_begin_end[1], :] = norm_feature[row_begin_end[0]: row_begin_end[1], :]

    return feature


# 构造原子特征
def compute_feature(mol, node_num: int, super_node: bool) -> np.array:
    """
    :param mol: 计算特征化学结构
    :param node_num: 化学分子原子数
    :param super_node: 是否加超级节点
    """
    atom_list = [5, 7, 6, 8, 9, 15, 16, 17, 35, 53]  # B N C O F P S Cl Br I
    if super_node:
        feature = np.zeros([node_num + 1, 30])
    else:
        feature = np.zeros([node_num, 30])

    for atom_num in range(0, node_num):
        atom = mol.GetAtomWithIdx(atom_num)
        # 0-10 相对原子质量
        try:
            feature[atom_num][atom_list.index(atom.GetAtomicNum())] = 1
        except:
            feature[atom_num][10] = 1

        feature[atom_num][len(atom.GetNeighbors()) + 11] = 1  # 11-16 邻域非氢原子个数
        feature[atom_num][atom.GetTotalNumHs() + 17] = 1  # 17-21 邻域氢原子个数
        feature[atom_num][22] = atom.GetFormalCharge()  # 22 电荷
        feature[atom_num][23] = atom.IsInRing()  # 23 Is in a ring
        feature[atom_num][24] = atom.GetIsAromatic()  # 24 Is aromatic
        """
        # 25原子参与形成化学键的平均值
        if len(atom.GetBonds()) == 0:
            feature[atom_num][1] = 0
        else:
            sum = 0
            for i in atom.GetBonds():
                sum += int(i.GetBondType())
            feature[atom_num][25] = sum / len(atom.GetBonds())
        """
    crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    i = 0
    for mol_log, mol_mr in crippen:
        feature[i][25] = mol_log  # 26a Crippen contribution to logP
        feature[i][26] = mol_mr  # 27a Crippen contribution to Molar Refractivity
        i += 1

    # 28a Total Polar Surface Area contribution
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
    for i, x in enumerate(tpsa):
        feature[i][27] = x

    # 29a Labute Approximate Surface Area contribution
    asa = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    for i, x in enumerate(asa):
        feature[i][28] = x

    # 30a Estate index
    estate = EState.EStateIndices(mol)
    for i, x in enumerate(estate):
        feature[i][29] = x

    """
    # 31a Gasteiger partial charge
    AllChem.ComputeGasteigerCharges(mol)
    for i in range(node_num):
        atom = mol.GetAtomWithIdx(i)
        feature[i][31] = atom.GetDoubleProp('_GasteigerCharge')
    """
    feature = normalization(feature)
    return feature


# smile化学式转换为邻接矩阵
def smile_to_graph(smile: str, miss_rate=0.5) -> np.array:
    """
       :param miss_rate: 原子特征的被遮掩率
       :param smile: str, 化学分子的smile化学式
       :return:
           np.array() 化学分子的邻接矩阵， 分子中原子的特征向量
    """
    # smile化学式转换为mol
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None, None, None, None, None, None, None

    Chem.SanitizeMol(mol)

    node_num = len(mol.GetAtoms())  # 节点个数
    edge_num = len(mol.GetBonds())  # 边个数

    edges = []  # 图边信息
    for i in range(edge_num):
        temp = []
        bond = mol.GetBondWithIdx(i)
        temp.append(bond.GetBeginAtomIdx())  # 第一个原子0
        temp.append(bond.GetEndAtomIdx())  # 第二个原子1
        # temp.append(int(bond.GetBondType()))  # 化学键信息
        temp.append(bond.GetBondTypeAsDouble())  # 化学键信息(双精度型)2
        temp.append(bond.GetIsAromatic())  # 是否芳香3
        temp.append(bond.GetIsConjugated())  # 是否共轭4
        temp.append(bond.IsInRing())  # 是否在环上5
        edges.append(temp)

    # 构造节点矩阵及边特征数组
    matrix = np.zeros([2, edge_num * 2 + node_num + node_num + 1], dtype=np.int64)
    edge_feature = np.zeros([edge_num * 2 + node_num + node_num + 1])
    edge_inform = np.zeros([edge_num * 2 + node_num + node_num + 1, 7])
    for i in range(edge_num):
        matrix[0, i * 2] = edges[i][0]
        matrix[1, i * 2] = edges[i][1]
        matrix[0, i * 2 + 1] = edges[i][1]
        matrix[1, i * 2 + 1] = edges[i][0]

        edge_feature[i * 2] = edges[i][2]
        edge_feature[i * 2 + 1] = edges[i][2]

        # 0-3 键特征
        if edges[i][2] == 1:
            edge_inform[i * 2, 0] = 1
            edge_inform[i * 2 + 1, 0] = 1
        elif edges[i][2] == 1.5:
            edge_inform[i * 2, 1] = 1
            edge_inform[i * 2 + 1, 1] = 1
        elif edges[i][2] == 2:
            edge_inform[i * 2, 2] = 1
            edge_inform[i * 2 + 1, 2] = 1
        else:
            edge_inform[i * 2, 3] = 1
            edge_inform[i * 2 + 1, 3] = 1
        # 4 是否芳香
        if edges[i][3] == 1:
            edge_inform[i * 2, 4] = 1
            edge_inform[i * 2 + 1, 4] = 1
        # 5 是否共轭
        if edges[i][4] == 1:
            edge_inform[i * 2, 5] = 1
            edge_inform[i * 2 + 1, 5] = 1
        # 6 是否在环上
        if edges[i][5] == 1:
            edge_inform[i * 2, 6] = 1
            edge_inform[i * 2 + 1, 6] = 1
    # 加入自循环
    for i in range(edge_num * 2, edge_num * 2 + node_num + 1):
        matrix[0, i] = i - edge_num * 2
        matrix[1, i] = i - edge_num * 2
        edge_feature[i] = 1
        edge_inform[i, 0] = 1

    # 构造原子端的特征向量
    feature = compute_feature(mol, node_num, super_node=True)

    # 全局超级节点
    for i in range(edge_num * 2 + node_num + 1, edge_num * 2 + node_num + node_num + 1):
        matrix[0, i] = i - edge_num * 2 - node_num - 1
        matrix[1, i] = node_num
        edge_feature[i] = 1
        edge_inform[i, 0] = 1

    # numpy转换成tensor
    matrix = torch.from_numpy(matrix)
    feature = torch.tensor(torch.from_numpy(feature), dtype=torch.float32)
    edge_feature = torch.tensor(torch.from_numpy(edge_feature), dtype=torch.float32)
    edge_inform = torch.tensor(torch.from_numpy(
        normalization(edge_inform, (0, edge_num * 2))),
        dtype=torch.float32)

    # 构造边的邻域索引(普通边+超级边)
    reversal_index = torch.tensor([], dtype=torch.int64)
    reversal_inform = torch.tensor([])
    for i in range(0, edge_num * 2, 2):
        reversal_inform = torch.cat((reversal_inform, edge_inform[i].view(1, -1)), dim=0)
        for j in range(i, edge_num * 2):
            if (matrix[0][j] == matrix[0][i] or matrix[0][j] == matrix[1][i]) and i != j and i + 1 != j:
                reversal_index = torch.cat((reversal_index,
                                            torch.tensor([[i // 2], [j // 2]], dtype=torch.int64),
                                            torch.tensor([[j // 2], [i // 2]], dtype=torch.int64)), dim=1)

    row_reversal_num = reversal_inform.shape[0]  # 不普通边个数
    reversal_inform = torch.cat((reversal_inform, edge_inform[edge_num * 2 + node_num + 1:
                                                              edge_num * 2 + node_num + 1 + node_num]), dim=0)
    for i in range(node_num):
        for j in range(0, edge_num * 2, 2):
            if matrix[0][j] == i or matrix[1][j] == i:
                reversal_index = torch.cat((reversal_index,
                                            torch.tensor([[j // 2], [row_reversal_num + i]], dtype=torch.int64)), dim=1)
    # 如果原始图没有正常边，将其全连接
    if reversal_index.shape[0] == 0:
        matrix = None
        """
        for i in range(reversal_inform.shape[0]):
            for j in range(i + 1, reversal_inform.shape[0]):
                reversal_index = torch.cat((reversal_index,
                                            torch.tensor([[i], [j]], dtype=torch.int64),
                                            torch.tensor([[j], [i]], dtype=torch.int64)), dim=1)
        """

    # 随机掩盖
    strong_feature = torch.clone(feature)
    mask = random.sample(range(0, feature.shape[0]), int(feature.shape[0] * miss_rate))
    strong_feature[mask] = 0.0

    return matrix, feature, edge_feature, edge_inform, edge_num * 2 + node_num + node_num + 1, \
           (reversal_inform, reversal_index, row_reversal_num), strong_feature


# 计算分子中的官能团
fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')


def compute_group(smile: str) -> np.array:
    """
    :param smile: 要计算官能团的化学式
    :return: 分子中的官能团数组(39个)
    group_num=torch.tensor(torch.from_numpy(compute_group(item['smile'])), dtype=torch.float32),
    """
    group = np.zeros(39)
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    fcat = FragmentCatalog.FragCatalog(fparams)
    fcgen = FragmentCatalog.FragCatGenerator()
    m = Chem.MolFromSmiles(smile)
    fcgen.AddFragsFromMol(m, fcat)
    num_entries = fcat.GetNumEntries()
    if num_entries == 0:
        return group

    l = list(fcat.GetEntryFuncGroupIds(num_entries - 1))
    if len(l) != 0:
        for i in l:
            group[i] += 1
    return group


# 计算子图特征(分解方法BRICS)
def compute_SubGraph(smile: str) -> dict:
    """
    :param smile 化学分子的smile化学式
    :return: 组合成的子图字典型
    SubGraph=compute_SubGraph(item['smile'])
    """
    m = Chem.MolFromSmiles(smile)
    frags = (BRICS.BRICSDecompose(m))

    matrix = torch.tensor([], dtype=torch.int64)
    node_feature = torch.tensor([])
    edge_feature = torch.tensor([])
    node_num = []
    node_sum = 0
    for fsmi in frags:
        a, b, c, d, e = smile_to_graph(fsmi)
        matrix = torch.cat((matrix, node_sum + torch.from_numpy(a)), dim=1)
        node_feature = torch.cat((node_feature, torch.tensor(torch.from_numpy(b), dtype=torch.float32)), dim=0)
        edge_feature = torch.cat((edge_feature, torch.tensor(torch.from_numpy(c), dtype=torch.float32)), dim=0)
        node_num.append(b.shape[0])
        node_sum += b.shape[0]

    subgraph = {'matrix': matrix, 'node_feature': node_feature, 'edge_feature': edge_feature, 'node_num': node_num}
    return subgraph


class DataSet(InMemoryDataset):
    def __init__(self, save_root, data_mode, miss_rate=0.5,
                 processed_file_names=None, dataset=None, transform=None, pre_transform=None):
        """
        :param miss_rate:原子特征的被遮掩率
        :param save_root:保存数据的目录
        :param pre_transform:在读取数据之前做一个数据预处理的操作
        :param transform:在访问之前动态转换数据对象
        :param data_mode:数据集的调用方式[train,test]
        """
        self.data_mode = data_mode
        self.miss_rate = miss_rate
        self.dataset = dataset
        self.processed_names = processed_file_names

        super(DataSet, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):  # 原始数据文件夹存放位置
        return ['']

    @property
    def processed_file_names(self):
        return [self.processed_names]

    def download(self):
        pass

    def process(self):
        if is_pass == 0:
            pass
        else:
            data = []
            datasets = []

            task = 0

            if self.dataset == 'TOX21':
                tasks, datasets, transformer = dc.molnet.load_tox21()

            elif self.dataset == 'BBBP':
                tasks, datasets, transformer = dc.molnet.load_bbbp()

            elif self.dataset == 'ClinTox':
                tasks, datasets, transformer = dc.molnet.load_clintox()

            elif self.dataset == 'BACE':
                tasks, datasets, transformer = dc.molnet.load_bace_classification()

            elif self.dataset == 'SIDER':
                tasks, datasets, transformer = dc.molnet.load_sider()

            elif self.dataset == 'PCBA':
                tasks, datasets, transformer = dc.molnet.load_pcba()

            elif self.dataset == 'HIV':
                tasks, datasets, transformer = dc.molnet.load_hiv()

            elif self.dataset == 'MUV':
                tasks, datasets, transformer = dc.molnet.load_muv()

            trainset, validset, testset = datasets

            if self.data_mode == 'train':
                for X, y, w, ids in trainset.itersamples():
                    data.append({'smile': ids, 'name': '1', 'label': int(y[task])})
            elif self.data_mode == 'test':
                for X, y, w, ids in testset.itersamples():
                    data.append({'smile': ids, 'name': '1', 'label': int(y[task])})

            data_list = []
            for item in data:
                edge_index, feature, edge_feature, edge_inform, edge_num, reversal, strong_feature = smile_to_graph(
                    item['smile'], miss_rate=self.miss_rate)

                if edge_index is None:
                    print('None')
                    continue

                data_list.append(Data(x=feature,
                                      edge_index=edge_index,
                                      edge_attr=edge_feature,
                                      edge_inform=edge_inform,
                                      edge_num=edge_num,
                                      reversal=reversal,
                                      strong_feature=strong_feature,
                                      simle=item['smile'],
                                      y=item['label']))

            data_save, data_slices = self.collate(data_list)
            torch.save((data_save, data_slices), self.processed_file_names[0])


# 保存增强数据
class StrongDataSet(InMemoryDataset):
    def __init__(self, save_root, data_mode,
                 processed_file_names=None, strong_data=None, transform=None, pre_transform=None):
        """
        :param save_root:保存数据的目录
        :param pre_transform:在读取数据之前做一个数据预处理的操作
        :param transform:在访问之前动态转换数据对象
        :param data_mode:数据集的调用方式[create, use]
        :param strong_data:增强后的数据
        """
        self.data_mode = data_mode
        self.strong_data = strong_data
        self.processed_names = processed_file_names

        super(StrongDataSet, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):  # 原始数据文件夹存放位置
        return ['']

    @property
    def processed_file_names(self):
        return [self.processed_names]

    def download(self):
        pass

    def process(self):
        if self.data_mode == 'use':
            pass
        elif self.data_mode == 'create' and self.strong_data is not None:
            data_save, data_slices = self.collate(self.strong_data)
            torch.save((data_save, data_slices), self.processed_file_names[0])
