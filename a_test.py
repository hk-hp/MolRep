import os
from torch_geometric.data import DataLoader
from pyg_DataSet import DataSet, StrongDataSet

batch_size = 1  # 批大小
datasets = ['TOX21', 'ClinTox', 'BBBP', 'SIDER']


def main():
    for dataset in datasets:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

        # 加载数据
        """
        data_set = torch.load(dataset + '/testAddNoiseData.pt')
        test_loader = Data.DataLoader(dataset=Data.TensorDataset(data_set['x'], data_set['y']),
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=2)
        """

        data_set = DataSet(save_root=dataset + '/matrix', data_mode='train', dataset=dataset,
                           processed_file_names=dataset + '/trainData.pt')
        # data_set = StrongDataSet(save_root=dataset + '/Strong', data_mode='use',
        #                         processed_file_names=dataset + '/trainStrong.pt')

        test_loader = DataLoader(data_set, batch_size=1, shuffle=True, num_workers=4)
        n = 0
        m = 0
        for data in test_loader:
            target = data.y
            m += 1
            n += target.item()
        print(dataset, m, n, m - n)




if __name__ == '__main__':
    main()

