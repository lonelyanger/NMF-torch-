from data.read_data import read_data
from data.incomplete_data import get_incomplete_data,get_sn
import warnings
import torch
from models import Multiview_network as MultiNet
from util.operations import Clustering,SpecClustering
warnings.filterwarnings("ignore")
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(5)


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    DSname='./datasets/handwritten.mat'
    DataSet=read_data(DSname)

    op=[]
    nSmp=DataSet.nSmp
    nClass=DataSet.nClass

    Mean_ACC = []
    Mean_NMI = []

    views_feadim=DataSet.views_feadim
    print("dimension for multiview:\n",views_feadim,"\n")
    print("shape of multiview data")
    for v in range(DataSet.view_num):
        print("view ",v," : ",DataSet.X[v].shape)

    model=MultiNet.MultiNMF(DataSet.view_num,DataSet.views_feadim,DataSet.nSmp,DataSet.nClass)
 

    NetInput=DataSet.X
    print("--------Pretrain Start!----------")
    epoch=400
    accmax=0
    for i in range(epoch):
        loss=model(NetInput,DataSet.view_num)
        # loss.backward()
        model.update_params()
        # output loss
        if i%20==0:
            H=model.getH().t().detach().numpy()
            print("it:",i)
           
            y_preds, scores, ACC, NMI = Clustering(H, DataSet.labels.ravel(), DataSet.nClass)
            if ACC>accmax:
                accmax=ACC
            print("loss is : ",loss.item())
     
    H=model.getH().t().detach().numpy()

    
    y_preds, scores, ACC, NMI = Clustering(H, DataSet.labels.ravel(), DataSet.nClass)
    if ACC>accmax:
        accmax=ACC
    print("loss is : ",loss.item())

    print("MAXACC is ",accmax)
    print("--------Pretrain End!----------")

    

