import matplotlib.pyplot as plt
import os
import torch
from torch_geometric.data import DataLoader
from pyg_DataSet import DataSet
from sklearn.metrics import roc_curve, auc

batch_size = 1  # 批大小
dataset = 'ClinTox'


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_set = DataSet(save_root=dataset + '/matrix', data_mode='test', processed_file_names=dataset + '/testData.pt',
                       dataset=dataset)
    test_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
    model = dataset + '/QY_GC' + str(54) + '.pkl'

    my_net = torch.load(model)
    # 测试模型
    my_net.eval()
    with torch.no_grad():  # 关闭梯度
        for data in test_loader:
            result, _ = my_net(data)
            _, prediction = result.max(dim=1)
            # target = data.y


def pic():
    times = 80

    filename = ['result/ClinTox.txt', 'result/SIDER.txt','result/TOX21.txt',
                'result/BBBP.txt']
    filename = ['result/ClinTox_train.txt', 'result/SIDER_train.txt','result/TOX21_train.txt',
                'result/BBBP_train.txt']
    labels = ['ClinTox', 'SIDER','TOX21', 'BBBP']
    data = []
    for path in filename:
        temp = []
        with open(path, 'r') as text:
            words = text.read().split('\n')
            for word in words:
                if word == '':
                    continue
                temp.append(float(word.split(':')[1][:4]))
        data.append(temp)

    ln1, = plt.plot(data[0][:times], color='red', linewidth=1.0, linestyle='-')
    ln2, = plt.plot(data[1][:times], color='blue', linewidth=1.0, linestyle='-')
    ln3, = plt.plot(data[2][:times], color='green', linewidth=1.0, linestyle='-')
    ln5, = plt.plot(data[3][:times], color='yellow', linewidth=1.0, linestyle='-')

    plt.legend(handles=[ln1, ln2, ln3, ln5], labels=labels)
    plt.title("Train")
    plt.show()


def is_in_list(num, l):
    for i in l:
        if num == i:
            return True
    return False


def simle():
    from rdkit.Chem import Draw
    from rdkit import Chem

    data_set = DataSet(save_root=dataset + '/matrix', data_mode='test', processed_file_names=dataset + '/testData.pt',
                       dataset=dataset)
    test_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
    data_num = 1
    for data in test_loader:
        smi = data.simle[0]
        mol = Chem.MolFromSmiles(smi)
        print(smi, 'result/&'+ str(data.y.item())+'pic' + str(data_num) + '.png')
        # print(len(mol.GetAtoms()))
        # (data_num)

        important = torch.load('result/' + str(data_num) + '.pt').view(-1)
        # print(important)
        index = []
        for num, i in enumerate(important[:-1]):
            if i >= 0.5:
                index.append(num)
        # print(index)

        hit_bonds = []
        for bond in mol.GetBonds():
            aid1 = bond.GetBeginAtomIdx()
            aid2 = bond.GetEndAtomIdx()
            if is_in_list(aid1, index) and is_in_list(aid2, index):
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())

        A = Draw.MolToImage(
            mol,
            subImgSize=(500, 500),
            highlightAtoms=index,
            highlightBonds=hit_bonds
        )
        # A.show()
        A.save('result/&'+ str(data.y.item())+'pic' + str(data_num) + '.png')
        data_num += 1


def box_pic():
    """
    :param data: 箱线图各个指标的一维数据
    :param label: 箱线图各个指标的名称
    """
    times = 80

    #filename = ['result/ClinTox.txt', 'result/去drop.txt', 'result/去GKC.txt',
    #            'result/去边.txt', 'result/去低频.txt', 'result/去高频.txt', 'result/去JL.txt']
    filename = ['result/BBBP.txt', 'result/BBBPqdrop.txt', 'result/BBBPqGKC.txt','result/BBBPqbian.txt',
                'result/BBBPqdi.txt', 'result/BBBPqgao.txt', 'result/BBBPqJL.txt']
    #filename = ['result/TOX21.txt', 'result/TOX21qdrop.txt', 'result/TOX21qGKC.txt','result/TOX21qbian.txt',
    #            'result/TOX21qdi.txt', 'result/TOX21qgao.txt', 'result/TOX21qJL.txt']
    labels = ['Original', 'Without Dropout', 'Without GKC', 'Without Edge', 'Without LF', 'Without HF', 'Without JK']
    data = []
    for path in filename:
        temp = []
        with open(path, 'r') as text:
            words = text.read().split('\n')
            for word in words:
                if word == '':
                    continue
                temp.append(float(word.split(':')[1][:4]))
        data.append(temp)

    # 绘图
    plt.boxplot(x=data,
                labels=labels,  # 添加具体的标签名称
                showmeans=True,
                patch_artist=True,
                boxprops={'color': 'black', 'facecolor': '#9999ff'},
                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
                meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
                medianprops={'linestyle': '--', 'color': 'orange'})

    plt.title("BBBP")
    # 显示图形
    plt.show()

if __name__ == '__main__':
    box_pic()
