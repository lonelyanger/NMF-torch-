import torch
import torch.nn as nn

class NMF(nn.Module):
    def __init__(self,view_num,views_feadim,nSmp,latdim,lr=0.001):
        super().__init__()
        # init latent represent
        self.USet=[]
        self.VSet=[]
        self.view_num=view_num
        
        for v in range(view_num):
            vfeadim=views_feadim[v]
            U=nn.Parameter(torch.rand(vfeadim,latdim),requires_grad=True)
            U=torch.softmax(U,dim=0)
            self.USet.append(U)
            
            V=nn.Parameter(torch.rand(latdim,nSmp),requires_grad=True)
            V=torch.softmax(V,dim=0)
            self.VSet.append(V)
        self.flag=-1

    def forward(self,X,view_num):
        output=dict()
        if self.flag==-1:
            self.flag=0
            self.X=X
        loss=dict()
        for v in range(view_num):
            loss_v=torch.norm(X[v].t()-torch.mm(self.USet[v],self.VSet[v]),p='fro')
            loss[v]=loss_v
        return loss

    def update_params(self):
        for v in range(self.view_num):
            U=self.USet[v]
            V=self.VSet[v]
            X=self.X[v].t().float()
            self.USet[v]=torch.mm(X,V.t())/torch.mm(U,torch.mm(V,V.t()))*U
            self.VSet[v]=torch.mm(U.t(),X)/torch.mm(U.t(),torch.mm(U,V))*V
        return
    def getH(self):
        return self.VSet

class MultiNMF(nn.Module):
    def __init__(self,view_num,views_feadim,nSmp,latdim,lr=0.001):
        super().__init__()
        self.USet=[]
        self.VSet=[]
        self.view_num=view_num
        V=torch.rand(latdim,nSmp)
        V=torch.softmax(V,dim=0)
        self.V=V
        for v in range(view_num):
            vfeadim=views_feadim[v]
            U=torch.rand(vfeadim,latdim)
            U=torch.softmax(U,dim=0)
            self.USet.append(U)

        self.flag=-1

    def forward(self,X,view_num):
        output=dict()
        if self.flag==-1:
            self.flag=0
            self.X=X
        loss=0
        for v in range(view_num):
            loss+=torch.norm(X[v].t()-torch.mm(self.USet[v],self.V),p='fro')/view_num
        return loss

    def update_params(self):
        V=self.V
        UX=torch.zeros(V.shape)
        UVV=torch.zeros(V.shape)
        for v in range(self.view_num):
            U=self.USet[v]
            X=self.X[v].t().float()
            self.USet[v]=torch.mm(X,V.t())/torch.mm(U,torch.mm(V,V.t()))*U
            UX+=torch.mm(U.t(),X)
            UVV+=torch.mm(U.t(),torch.mm(U,V))
        self.V=UX/UVV*V
        return
    def getH(self):
        return self.V