import torch
from pyg_DataSet import StrongDataSet
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

add_times = 0  # 加扰次数
two_dim_add_times = 5
dataset = 'BBBP'
mode = 'train'
# mode = 'test'

def AddNoise(mean=0, var=0.5):
    # 加载数据
    data_set = StrongDataSet(save_root=dataset + '/Strong', data_mode='use',
                             processed_file_names=dataset + '/' + mode + 'Strong.pt')

    torch_data = torch.zeros(len(data_set), 128, dtype=torch.float32)
    label = torch.zeros(len(data_set), dtype=torch.long)
    for num, data in enumerate(data_set):
        torch_data[num] = data.super.squeeze()
        label[num] = data.y

    AddNoiseData = torch_data.clone()
    AddNoiseLabel = label.clone()

    matrix_data = torch.tensor([])
    label_data = torch.tensor([], dtype=torch.long)
    if mode == 'train':
        for i in range(add_times):
            """
            noise = np.random.normal(mean, var ** 0.5, torch_data.shape)
            for num, data in enumerate(torch_data):
                mask = random.sample(range(0, 128), int(128 * 0.8))
                temp = data.clone()
                temp[mask] = 0.0
                AddNoiseData = torch.cat((AddNoiseData, torch.tensor(temp + noise[num], dtype=torch.float32).view(1, -1)), 0)
                #AddNoiseData = torch.cat((AddNoiseData, temp.view(1, -1)), 0)
            AddNoiseLabel = torch.cat((AddNoiseLabel, label))
            """

            noise = np.random.normal(mean, var ** 0.5, torch_data.shape)
            AddNoiseData = torch.cat((AddNoiseData, torch.tensor(torch_data + noise, dtype=torch.float32)), 0)
            AddNoiseLabel = torch.cat((AddNoiseLabel, label))

        for num, i in enumerate(AddNoiseData):
            matrix = i[0:32].view(-1, 1).mm(i[0:32].view(1, -1)).unsqueeze(dim=0).unsqueeze(dim=0)
            matrix_data = torch.cat((matrix_data,
                                         matrix,), )
            label_data = torch.cat((label_data,
                                    AddNoiseLabel[num].view(1)), )

        for num, i in enumerate(AddNoiseData):
            matrix = i[0:32].view(-1, 1).mm(i[0:32].view(1, -1)).unsqueeze(dim=0).unsqueeze(dim=0)
            # select = random.randint(0, 1)
            # if select == 0:
            for j in range(two_dim_add_times):
                noise = np.random.normal(matrix.mean(), matrix.std()*4, matrix.shape)
                noise1 = np.random.normal(matrix.mean(), matrix.std()*6, matrix.shape)
                matrix_data = torch.cat((matrix_data,
                                         torch.tensor(matrix + noise1, dtype=torch.float32),
                                         # matrix.flip(dims=(3,)),
                                         torch.tensor(matrix + noise, dtype=torch.float32)), )
                label_data = torch.cat((label_data,
                                        AddNoiseLabel[num].view(1),
                                        # AddNoiseLabel[num].view(1),
                                        AddNoiseLabel[num].view(1)), )
    else:
        for num, i in enumerate(AddNoiseData):
            matrix = i[0:32].view(-1, 1).mm(i[0:32].view(1, -1)).unsqueeze(dim=0).unsqueeze(dim=0)
            matrix_data = torch.cat((matrix_data, matrix), )
        label_data = AddNoiseLabel

    # create_pic(matrix_data, label_data)
    data = {'x': matrix_data, 'y': label_data}
    torch.save(data, dataset + '/' + mode + 'AddNoiseData.pt')


def create_pic(data, label):
    data = np.array(data.squeeze(1))
    for num, i in enumerate(data):
        ax = sns.heatmap(i, annot=False, center=0)
        plt.savefig('BBBP/B/' + str(label[num].item()) + 'Train' + str(num) + '.jpg')
        plt.close()


if __name__ == '__main__':
    AddNoise()
