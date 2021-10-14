import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class MNet_RecOnly(nn.Module):
    def __init__(self,view_num,views_feadim,nSmp,latdim,lr=0.001):
        super().__init__()
        # init latent represent
        self.H=nn.Parameter(torch.rand(nSmp,latdim),requires_grad=True)
        nn.init.xavier_uniform_(self.H)
        self.extdim=100
        # setting decoder
        self.Decoder=[]
        self.Opt=[]
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.vfeadim=torch.tensor(views_feadim)
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.6)
            d=nn.Sequential(
                nn.Linear(latdim,latdim),
                nn.Linear(latdim,self.extdim),
                nn.ReLU(),
                # nn.BatchNorm1d(self.extdim),

                nn.Linear(self.extdim,vfeadim2),
                nn.ReLU(),
                # nn.BatchNorm1d(vfeadim2),

                nn.Linear(vfeadim2,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,vfeadim)
            )
            for m in d:
                if isinstance(m,nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))

    def forward(self,X,Sn,view_num):
        output=dict()
        loss=0
        for v in range(view_num):
            output[v]=self.Decoder[v](self.H)
            loss+=torch.norm((output[v]-X[v]).t()*Sn[:,v])
            
            # print(loss.item())
        return loss

    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()

    def getH(self):
        return self.H


class Weighted_Fusion(nn.Module):
    def __init__(self,Sn,view_num,latdim):
        super(Weighted_Fusion,self).__init__()
        self.weight=Sn
        self.view_num=view_num
        self.latdim=latdim

    def forward(self,Hv):
        h=0
        latdim=self.latdim
        for v in range(self.view_num):
            h+=torch.mul(Hv[v].t(),self.weight[:, v])
        fusion = torch.div(h, torch.sum(self.weight.float(), dim=1)).t()
        return fusion
    
class SelfAttention_Fusion(nn.Module):
    def __init__(self,Sn,view_num,latdim,nSmp,lr=0.0005):
        super(SelfAttention_Fusion,self).__init__()
        self.weight=Sn
        self.view_num=view_num
        self.latdim=latdim
        self.att=nn.Sequential(
            nn.Linear(in_features=latdim,out_features=latdim,bias=False),
            nn.Linear(in_features=latdim,out_features=1,bias=False)
        )
        for m in self.att:
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        self.Softmax=nn.Softmax()
        self.Opt=torch.optim.Adam(self.att.parameters(),lr=lr)

    def forward(self,Hv):
        att=0
        latdim=self.latdim
        for v in range(self.view_num):
            hv=torch.mul(Hv[v].t(),self.weight[:, v]).t()
            attv=self.att(hv)
            if isinstance(att,int):
                att=attv
            else :
                att=torch.cat((att,attv),1)
        att=torch.div(att,torch.sqrt(torch.tensor(latdim)))
        att=torch.where(att==0.0,-1*np.inf,att)
        att=self.Softmax(att)
        self.AttWeight=att
        h=0
        for v in range(self.view_num):
            h+=torch.mul(Hv[v].t(),att[:, v])
        # fusion = torch.
        return h.t()

class SelfAttLayer(nn.Module):
    def __init__(self,Sn,view_num,latdim,nSmp,lr=0.0005):
        super(SelfAttLayer,self).__init__()
        Sn=0

    def forward(self,Hv):
        return

class MNet_AE(nn.Module):
    def __init__(self,view_num, views_feadim, nSmp,latdim,Sn,lr=0.005,fusiontype=1):
        super().__init__()
        self.Opt=[]   
        self.extdim=100
        self.vfeadim=torch.tensor(views_feadim)
        self.Encoder=[]
        self.Decoder=[]
        

        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.6)
            e=nn.Sequential(
                nn.Linear(vfeadim,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,self.extdim),
                nn.ReLU(),
            )
             
            d=nn.Sequential(
                nn.Linear(latdim,latdim),
                nn.Linear(latdim,self.extdim),
                nn.ReLU(),
                # nn.BatchNorm1d(self.extdim),

                nn.Linear(self.extdim,vfeadim2),
                nn.ReLU(),
                # nn.BatchNorm1d(vfeadim2),

                nn.Linear(vfeadim2,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,vfeadim)
            )
            for m in d:
                if isinstance(m,nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
       
    
    def forward(self,X,Sn,view_num):
       return

    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
        
    def getH(self):
        return self.H

class MCapNet(nn.Module):
    def __init__(self,view_num,views_feadim,nSmp,latdim):
        super().__init__()
        # init latent represent
        self.H=nn.Parameter(torch.rand(nSmp,latdim))
        # setting decoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            d=nn.Sequential(
                nn.Linear(latdim,1500),
                nn.ReLU(),
                nn.Linear(1500,vfeadim*0.8),
                nn.ReLU(),
                nn.Linear(vfeadim*0.8,vfeadim*0.8),
                nn.ReLU(),
                nn.Linear(vfeadim*0.8,vfeadim)
            )
            self.Decoder.append(d)
    
    def forward(self,x,Sn):
        return x
    
class MNet_RecKL(nn.Module):
    def __init__(self,view_num,views_feadim,nSmp,latdim,lr=0.005):
        super().__init__()
        # list of Optim
        self.Opt=[]

        # init latent represent
        self.H=nn.Parameter(torch.rand(nSmp,latdim),requires_grad=True)
        nn.init.xavier_uniform_(self.H)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))

        # init clustering layer
        self.Clustering=Clustering(latdim,latdim)
        self.Opt.append(torch.optim.Adam(self.Clustering.parameters(),lr=lr))
        self.KLloss=torch.nn.KLDivLoss(size_average=False,reduce=True)
        # setting decoder
        self.Decoder=[]
        self.extdim=100
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.6)
            d=nn.Sequential(
                nn.Linear(latdim,self.extdim),
                nn.ReLU(),
                # nn.BatchNorm1d(self.extdim),

                nn.Linear(self.extdim,vfeadim2),
                nn.ReLU(),
                # nn.BatchNorm1d(vfeadim2),

                nn.Linear(vfeadim2,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,vfeadim)
            )
            for m in d:
                if isinstance(m,nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr/5))

    def forward(self,X,Sn,view_num,target_i,Mode="Pretrain"):
        output=dict()
        loss=0
        if Mode=="Pretrain":
            for v in range(view_num):
                output[v]=self.Decoder[v](self.H)
                loss+=torch.norm((output[v]-X[v]).t()*Sn[:,v])
        else:
            loss_q,q=self.Clustering(self.H)
            loss+=self.KLloss(loss_q,target_i)
            for v in range(view_num):
                output[v]=self.Decoder[v](self.H)
                loss+=torch.norm((output[v]-X[v]).t()*Sn[:,v])
        return loss
        
    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()

    def getH(self):
        return self.H

class Attention_Fusion(nn.Module):
    def __init__(self,view_num,latdim,nSmp,lr=0.0005):
        super(Attention_Fusion,self).__init__()
        self.view_num=view_num
        self.latdim=latdim
        self.Q=nn.Linear(in_features=latdim,out_features=latdim,bias=False)
        nn.init.xavier_uniform_(self.Q.weight)
        self.Softmax=nn.Softmax()
        self.Opt=torch.optim.Adam(self.Q.parameters(),lr=lr)

    def forward(self,Hv,H):
        w=torch.zeros(self.view_num)
        latdim=self.latdim
        for v in range(self.view_num):
            HQ=self.Q(Hv[v])
            attv=torch.div(torch.sum(torch.mul(H,HQ),1),torch.sqrt(torch.tensor(self.latdim)))
            w[v]=torch.sum(attv)
        w=self.Softmax(w)
        self.AttWeight=w
        
        return w

class MNet_RecCS(nn.Module):
    def __init__(self,view_num,views_feadim,nSmp,latdim,W,lr=0.005):
        super().__init__()
        # graph init
        self.constructL(W)
        
        # list of Optim
        self.Opt=[]

        # init latent represent
        self.H=nn.Parameter(torch.rand(nSmp,latdim),requires_grad=True)
        nn.init.xavier_uniform_(self.H)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))

        self.weighted_fusion=Attention_Fusion(view_num,latdim,nSmp,lr=lr)
        self.Opt.append(self.weighted_fusion.Opt)
        
        # setting decoder
        self.Decoder=[]
        self.Latv=[]
        self.extdim=100
        self.vfeadim=torch.tensor(views_feadim)
        for v in range(view_num):
            latv=nn.Sequential(
                nn.Linear(latdim,int(latdim*0.8)),
                nn.Linear(latdim,latdim))
            self.Latv.append(latv)
            self.Opt.append(torch.optim.Adam(latv.parameters(),lr=lr))
            nn.init.xavier_uniform_(latv.weight)

            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.6)
            d=nn.Sequential(
                nn.Linear(latdim,self.extdim),
                nn.ReLU(),

                nn.Linear(self.extdim,vfeadim2),
                nn.ReLU(),

                nn.Linear(vfeadim2,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,vfeadim)
            )
            for m in d:
                if isinstance(m,nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
         
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
    def constructL(self,W):
        self.G=[]
        self.L=[]
        view_num=len(W)
        for v in range(view_num):
            alpha=torch.mean(torch.tensor(W[v]))
            g=torch.exp(-1*torch.tensor(W[v])/(2*alpha))
            self.G.append(g)
            D=torch.diag(torch.sum(g,dim=0))
            self.L.append(D-g)
            
    
    def forward(self,X,Sn,view_num,Mode="Pretrain"):
        output=dict()
        Hv=dict()
        loss=0
        for v in range(view_num):
            Hv[v]=self.Latv[v](self.H)
            output[v]=self.Decoder[v](Hv[v])
        if Mode=="Pretrain":
            G=torch.zeros(self.G[0].shape)
            for v in range(view_num):
                loss+=torch.norm((output[v]-X[v]).t()*Sn[:,v])
                # G+=self.L[v]
            # HGH=torch.mm(self.H.t(),torch.mm(G,self.H))
            # loss+=torch.trace(HGH)
        else:
            w=self.weighted_fusion(Hv,self.H)
            # w=torch.ones(view_num)
            G=torch.zeros(self.G[0].shape)
            for v in range(view_num):
                loss+=torch.norm((output[v]-X[v]).t()*Sn[:,v])
                G+=self.G[v]*w[v]
            # CSloss
            HGH=torch.mm(self.H.t(),torch.mm(G,self.H))
            HI=torch.diag(HGH)
            HGH=HGH-torch.diag_embed(HI)
            HI=torch.reshape(HI,(1,len(HI)))
            loss+=torch.sum(torch.triu(torch.div(HGH,torch.mm(HI.t(),HI))))
           
            # print(loss.item())
        return loss
        
    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()

    def getH(self):
        return self.H