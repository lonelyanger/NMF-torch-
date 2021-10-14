import scipy.io as sio
import numpy as np
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
torch.set_default_tensor_type(torch.FloatTensor)


class MultiDataSet(Dataset):
    def __init__(self, data, view_number, labels):
        """
        Construct a DataSet.
        """
        self.X = dict()
        self.nSmp = data[0].shape[0]
        self.labels = labels
        self.view_num=view_number
        self.nClass=len(np.unique(labels))
        for v_num in range(view_number):
            self.X[v_num] = torch.tensor(data[v_num])

    def __getitem__(self, idx):
        x=[]
        for key in self.X.keys():
            x.append(self.X[key][idx])
        return x,self.labels[idx]
       
    @property
    def views_feadim(self):
        dim=[]
        for v_num in range(self.view_num):
            dim.append(self.X[v_num].shape[1])
        return dim
    # @property
    # def labels(self):
    #     return self.labels

    # @property
    # def num_examples(self):
    #     return self._num_examples

def Normalize(data):
    """
    :param data:Input datasets
    :return:normalized datasets
    """
    # m = np.mean(data)
    # mx = np.max(data)
    # mn = np.min(data)
    # return (data - m) / (mx - mn)
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    # return data/(mx+0.0001)
    return (data - m) / (mx - mn)

def read_data(str_name, Normal=1):
    """
    :param str_name:path and dataname
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    data = sio.loadmat(str_name)
    view_number = data['X'].shape[1]
    X=data['X'][0]
    X_train = []
    if min(data['gt']) == 0:
        labels = data['gt'] + 1
    else:
        labels = data['gt']
    for v_num in range(view_number):
        Xv=X[v_num].transpose()
        if (Normal == 1):
            Xv=Normalize(Xv)
        X_train.append(Xv)
    completedata = MultiDataSet(X_train, view_number, np.array(labels))
    return completedata

if __name__=="__main__":
    file_name='datasets/handwritten.mat'
    DataSet=read_data(file_name)
    print(DataSet.labels.shape)
    print(DataSet.nSmp)
    print(DataSet.view_num)
    print(DataSet.X[0])
    print(DataSet.X[1].shape)
    print(DataSet.views_feadim)
    print("test read_data successful")
