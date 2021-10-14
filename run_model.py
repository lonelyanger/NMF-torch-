from data.read_data import read_data
from data.incomplete_data import get_incomplete_data,get_sn,get_knn_affinity
import warnings
import torch
from util.operations import clustering_metric
from models import Multiview_Simple_network as MultiNet
from models import Multiview_Simple_TensorNet as MultiTSNet
from util.operations import Clustering,SpecClustering
from sklearn.cluster import KMeans
from scipy.io import savemat
warnings.filterwarnings("ignore")
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(5)


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    DSname='./datasets/handwritten.mat'
    DataSet=read_data(DSname)
    miss_rate=0.3
    Sn=get_sn(DataSet.view_num,DataSet.nSmp,miss_rate)
    DataSet=get_incomplete_data(DataSet,Sn)
    
    # Sn=torch.zeros(Sn.shape)
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

    
    W=get_knn_affinity(DataSet.X,DataSet.view_num,DataSet.nSmp,Sn)
    Sn=torch.tensor(Sn)

    # model=MultiNet.MNet_RecOnly(DataSet.view_num,DataSet.views_feadim,DataSet.nSmp,DataSet.nClass)
    model=MultiNet.MNet_AE(DataSet.view_num,DataSet.views_feadim,DataSet.nSmp,DataSet.nClass,Sn,W)
    # model=MultiNet.MNet_Rec_Att_Graph_MultiSubspace(DataSet.view_num,DataSet.views_feadim,DataSet.nSmp,DataSet.nClass,W)
    # model=MultiNet.MNet_Rec_CS(DataSet.view_num,DataSet.views_feadim,DataSet.nSmp,DataSet.nClass,W)
    # model=MultiNet.MNet_Rec_Gra(DataSet.view_num,DataSet.views_feadim,DataSet.nSmp,DataSet.nClass,W)
    # model=MultiNet.MNet_Rec_Gra_Sub(DataSet.view_num,DataSet.views_feadim,DataSet.nSmp,DataSet.nClass,W)
    # model=MultiTSNet.MNet_Tensor(DataSet.view_num,DataSet.views_feadim,DataSet.nSmp,DataSet.nClass,W,Sn)
    model.to(device)

    NetInput=DataSet.X
    print("--------Pretrain Start!----------")
    epoch=200
    accmax=0.8
    for i in range(epoch):
        loss=model(NetInput,Sn,DataSet.view_num,"Pretrain")
        loss.backward()
        model.update_params()
        # output loss
        if i%100==0:
            H=model.getH().detach().numpy()
            
            y_preds, scores, ACC, NMI = Clustering(H, DataSet.labels.ravel(), DataSet.nClass)
            # y_preds, scores, ACC, NMI = SpecClustering(H, DataSet.labels.ravel(), DataSet.nClass)
            # if ACC>accmax:
            #     # accmax=ACC
            #     y_preds, scores, ACC, NMI = SpecClustering(H, DataSet.labels.ravel(), DataSet.nClass)
            print("it:",i,". loss is : ",loss.item())
     
       
    H=model.getH().detach().numpy()
    y_preds, scores, ACC, NMI = Clustering(H, DataSet.labels.ravel(), DataSet.nClass)
    # y_preds1, scores1, ACC1, NMI1= SpecClustering(H,DataSet.labels.ravel(), DataSet.nClass)  
    print("--------Pretrain End!----------")


    print("--------Fintune Start!----------")
    epoch=100
    for i in range(epoch):
        loss=model(NetInput,Sn,DataSet.view_num,"Train")
        loss.backward()
        model.update_params()
        if i%10==0:
            H=model.getH().detach().numpy()
            
            y_preds, scores, ACC, NMI = Clustering(H, DataSet.labels.ravel(), DataSet.nClass)
            # y_preds, scores, ACC, NMI = SpecClustering(H, DataSet.labels.ravel(), DataSet.nClass)
            # if ACC>accmax:
            #     # accmax=ACC
            #     y_preds, scores, ACC, NMI = SpecClustering(H, DataSet.labels.ravel(), DataSet.nClass)
            print("it:",i,". loss is : ",loss.item())
     
    # H=model.getH().detach().numpy()
    
    H=model.getH().detach()
    # H=torch.where(H<=torch.FloatTensor([0.1]),torch.FloatTensor([0.0]),H)
    H=H.numpy()
    # L=model.comL.detach().numpy()
    print(H)
    val,idx=torch.max(model.getH().detach(),1)
    print(idx)
    scores, acc, nmi = clustering_metric(DataSet.labels.ravel(), idx.numpy(), DataSet.nClass)
    
    y_preds, scores, ACC, NMI = Clustering(H, DataSet.labels.ravel(), DataSet.nClass)
    # y_preds1, scores1, ACC1, NMI1= SpecClustering(H,DataSet.labels.ravel(), DataSet.nClass)  
    print("--------Fintune End!----------")
    print(accmax)
    savemat("H.mat",{"H":H})
    # savemat("L.mat",{"L":L})

