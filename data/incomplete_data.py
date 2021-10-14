import numpy as np
from numpy.random import randint
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import kneighbors_graph
import scipy.io as sio
import torch
torch.set_default_tensor_type(torch.FloatTensor)
np.random.seed(5)
def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete datasets information, simulate partial view datasets with complete view datasets
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """
    one_rate = 1-missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len,1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix

def get_incomplete_data(DS, Sn):
    view_num=DS.view_num
    nSmp=DS.nSmp
    for i in range(view_num):
        for j in range(nSmp):
            if Sn[j][i]==0:
                DS.X[i][j,:]=0
    #imp = SimpleImputer(missing_values='NaN', strategy='mean')
    #for j in range(Sn.shape[1]):
        #imp.fit(complete_data.data[str(j)])

    return DS

def get_knn_affinity(input_data, view_num, N_samples, Sn):
    W = []
    for i in range(view_num):
        temp = Sn[:, i]-np.ones(N_samples)
        zero_index = np.nonzero(temp)
        X = np.delete(input_data[i], zero_index[0], axis=0)
        G = np.zeros((X.shape[0], N_samples))
        # nozero_index = (np.nonzero(Sn[:, i])).t()
        # print('nozero_index_shape:', nozero_index[0].shape)
        # for index in range(len(nozero_index[0])):
        #     G[index][nozero_index[0][index]] = 1
        nozero_index=np.nonzero(Sn[:,i])[0].transpose()
        # print('nozero_index_shape:', nozero_index.shape)
        for index in range(len(nozero_index)):
            G[index][nozero_index[index]] = 1
        n_neighbors=5
        # n_neighbors=X.shape[0]-1
        # A = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False)
        A = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
        A = A.toarray()
        
        W.append(np.dot(np.dot(G.T, A), G))
        # print('W:', W[i].shape)
    return W



if __name__=="__main__":
    import read_data
    file_name='datasets/handwritten.mat'
    DataSet=read_data.read_data(file_name)
    miss_rate=0.3
    Sn=get_sn(DataSet.view_num,DataSet.nSmp,miss_rate)
    DataSet=get_incomplete_data(DataSet,Sn)
    print(Sn)
    print(DataSet.X[0])
    Wn=get_knn_affinity(DataSet.X,DataSet.view_num,DataSet.nSmp,Sn)
    print("test incompletedata successful")