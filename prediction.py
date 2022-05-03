import os
import torch
import math
from torch_geometric.data import DataLoader
from pyg_DataSet import DataSet, StrongDataSet
import global_value
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

batch_size = 1  # 批大小
dataset = 'SIDER'


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

    # 加载数据
    """
    data_set = torch.load(dataset + '/testAddNoiseData.pt')
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(data_set['x'], data_set['y']),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2)
    """

    data_set = DataSet(save_root=dataset + '/matrix', data_mode='test',  processed_file_names=dataset + '/testData.pt')
    #data_set = StrongDataSet(save_root=dataset + '/Strong', data_mode='use',
    #                         processed_file_names=dataset + '/trainStrong.pt')

    test_loader = DataLoader(data_set, batch_size=1, shuffle=True, num_workers=4)

    model = []
    for i in range(140):
        model.append(dataset + '/3' + str(i) + '.pkl')

    for i in model:
        my_net = torch.load(i)
        # 测试模型
        test_correct = 0
        test_number = 0
        my_net.eval()
        y_true = []
        y_score = []
        y_prediction = []

        with torch.no_grad():  # 关闭梯度
            for data in test_loader:
                # result = my_net(data[0])
                result, _ = my_net(data)
                _, prediction = result.max(dim=1)
                # target = data[1]
                target = data.y
                # print(target.item(), prediction.item())

                # y_true.append(data[1].item())
                y_true.append(data.y.item())
                y_score.append(result[0, 1].item())
                y_prediction.append(prediction.item())

                test_correct += (prediction == target).sum().item()
                test_number += target.size(0)


        #print("Accuracy of Test Samples:{}%".format(100 * test_correct / test_number))

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        print(i + "  AUC:{}%".format(auc(fpr, tpr) * 100))
        #print("F1:{}%".format(f1_score(y_true, y_prediction, average='weighted')))



if __name__ == '__main__':
    main()
