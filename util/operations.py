from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, cluster
from sklearn.neighbors import kneighbors_graph
import numpy as np
import torch
from munkres import Munkres
from sklearn.cluster import KMeans, SpectralClustering
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
def EuDist(X,Y):
    # X nSmp_x*nFeX
    # Y nSmp_y*nFeX
    # D nSmp_x*nSmp_y
    XX = X**2
    YY = Y**2
    XY=torch.mm(X,Y.t())
    XX = torch.sum(XX,dim=1).unsqueeze(1) 
    YY = torch.sum(YY,dim=1).unsqueeze(0)  
    DD=XX+YY-2*XY
    DD=torch.where(DD<torch.zeros(1),torch.zeros(1),DD)
    D=torch.sqrt(DD)
    return D
    
def get_knn_affinity(input_data, view_num, N_samples, Sn):
    W = []
    for i in range(view_num):
        temp = Sn[:, i]-np.ones(N_samples)
        zero_index = np.nonzero(temp)
        X = np.delete(input_data.data[str(i)], zero_index[0], axis=0)
        G = np.zeros((X.shape[0], N_samples))
        nozero_index = np.nonzero(Sn[:, i])
        print('nozero_index_shape:', nozero_index[0].shape)
        for index in range(len(nozero_index[0])):
            G[index][nozero_index[0][index]] = 1
        A = kneighbors_graph(X, 15, mode='distance', include_self=False)
        A = A.toarray()
        W.append(np.dot(np.dot(G.T, A), G))
        print('W:', W[i].shape)
    return W

def Orthogonal_op(x):
    '''
    Computes a matrix that orthogonalizes the input matrix x

    x:      an n x d input matrix
    eps:    epsilon to prevent nonzero values in the diagonal entries of x

    returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                right multiplication
    '''

    x_2 = torch.mm(x.t(), x)
    e = torch.Tensor([0.00001])
    epsilon = e.to(DEVICE)[0]
    temp = torch.eye(x.shape[1]).to(DEVICE)
    x_2 += temp*epsilon
    #print(x_2)
    L = torch.cholesky(x_2)
    # 最后乘以的根号数字是什么
    y = torch.tensor(x.shape[0])
    y = y.float()
    ortho_weights = (torch.inverse(L)).t() * torch.sqrt(y)

    return ortho_weights.float()

def squared_distance(X, Y=None):
    if Y is None:
        Y = X
    # distance = squaredDistance(X, Y)
    # K.ndim(X)返回X的轴数
    sum_dimensions = list(range(2, X.ndim + 1))
    # 维数扩充 axis=1表示扩充为nX1Xm维 axis=0表示扩充为1XnXm维
    X = torch.unsqueeze(X, dim=1)
    # 利用广播机制
    squared_difference = torch.square(X - Y)
    distance = torch.sum(squared_difference, axis=sum_dimensions[0])
    return distance.float()

def pairwise_distance(X, Y):
    '''
    Calculates the pairwise distance between points in X and Y
    '''
    squared_difference = torch.square(X - Y)
    distance = torch.sum(squared_difference, dim=-1)
    #return distance.float()
    return distance
def SpecClustering(x, y, num_clusters):
    # a = SpectralClustering(n_clusters=num_clusters, random_state=1, affinity='precomputed', n_init=30)
    # pred = a.fit_predict(x)
    pred = SpectralClustering(n_clusters=num_clusters, random_state=10,  n_init=30).fit_predict(x)
    y_preds = get_y_preds( y,pred, num_clusters)
    if np.min(y) == 1:
        y = y - 1
    scores,acc,nmi = clustering_metric(y, pred, num_clusters)

    ret = {}
    ret['kmeans'] = scores
    return y_preds, ret,acc,nmi
def Clustering(x_list, y, num_clusters):
    print('******** Clustering ********')
    kmeans = KMeans(n_clusters=num_clusters, random_state=10, max_iter=30).fit(x_list)
    kmeans_assignments = kmeans.labels_
    y_preds = get_y_preds(y, kmeans_assignments, num_clusters)
    if np.min(y) == 1:
        y = y-1
    scores, acc, nmi = clustering_metric(y, kmeans_assignments, num_clusters)
    ret = {}
    ret['kmeans'] = scores
    return y_preds, ret, acc, nmi

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:,j]) # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i,j]
            cost_matrix[j,i] = s-t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments)!=0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # recall
    recall = metrics.recall_score(y_true, y_pred, average=average)
    recall = np.round(recall, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    if verbose:
        # print('Confusion Matrix')
        # print(confusion_matrix)
        print('accuracy', accuracy, 'precision', precision, 'recall', recall, 'f_measure', f_score)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix



def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)

    # classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)
    # 下列AMI NMI ARI是聚类度量方法，可以不考虑聚类标签的实际值
    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)
    # ACC 计算accuracy是分类度量  因此需要将聚类标签与实际标签先作对应
    accuracy = metrics.accuracy_score(y_true, y_pred_ajusted)
    accuracy = np.round(accuracy, decimals)

    if verbose:
        print('ACC', accuracy, 'AMI', ami, 'NMI:', nmi, 'ARI:', ari)
    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}),accuracy,nmi

def getLaplacian(W):
        """input matrix W=(w_ij)
        "compute D=diag(d1,...dn)
        "and L=D-W
        "and Lbar=D^(-1/2)LD^(-1/2)
        "return Lbar
        """
        d=torch.sum(W,dim=0)
        D=torch.diag(d)
        L=D-W
        #Dn=D^(-1/2)
        Dn=torch.sqrt(torch.matrix_power(D,-1))
        Lbar=torch.mm(torch.mm(Dn,L),Dn)
        return Lbar
def getEigVec(Lbar, k):
    """input
    "matrix Lbar and k
    "return
    "k smallest eigen values and their corresponding eigen vectors
    """
    Lbar =Lbar.detach().numpy()
    eigval, eigvec = np.linalg.eig(Lbar)
    dim = len(eigval)

    # 查找前k小的eigval
    dictEigval = dict(zip(eigval, range(0, dim)))
    kEig = np.sort(eigval)[0:k]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix], torch.from_numpy(eigvec[:, ix])

def Normalize(data):
    """
    :param data:Input datasets
    :return:normalized datasets
    """
    # 按行求和
    m = data*data
    mx = np.maximum(np.sqrt(np.sum(m, axis=0)), 1e-8)
    out = np.true_divide(data, mx)
    return out

def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)

    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp

def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs,0)
    for i in range(N):
        Cabs[:,i]= Cabs[:,i] / (Cabs[ind[0,i],i] + 1e-6)
    Cksym = Cabs + Cabs.T;
    return Cksym

def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    nmi = normalized_mutual_info_score(gt_s[:], c_x[:])
    ari = adjusted_rand_score(gt_s[:], c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate ,nmi, ari

def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C,axis=0)
    W = np.diag(1.0/W)
    L = W.dot(C)
    return L

def form_structure_matrix(idx, K):
    Q = np.zeros((len(idx), K))
    for i, j in enumerate(idx):
        Q[i,j-1] = 1
    return Q

def form_Theta(Q):
    Theta = np.zeros((Q.shape[0], Q.shape[0]))
    for i in range(Q.shape[0]):
        Qq = np.tile(Q[i], [Q.shape[0],1])
        Theta[i, :] = 1/2*np.sum(np.square(Q - Qq), 1)
    return Theta

def get_one_hot_Label(Label):
        if Label.min()==0:
            Label = Label
        else:
            Label = Label - 1

        Label = np.array(Label)
        n_class = Label.max()
        n_sample = Label.shape[0]
        one_hot_Label = np.zeros((n_sample, n_class))
        for i,j in enumerate(Label):
            one_hot_Label[i, j] = 1

        return one_hot_Label

