
import random,torch,os,sys,ot,time,glob
import numpy as np
from multiprocessing import Pool
from numpy import array,arange,ones
tmp_cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(tmp_cfd)
from functools import reduce
import networkx as nx
from networkx.algorithms import shortest_paths
from sklearn.manifold import MDS,SpectralEmbedding
from myRWD_spring import myMultiEmdRWD
import umap
from node2vec import Node2Vec
import pickle
from scipy.sparse import csgraph


class Node:
    def __init__(self, global_identity):
        self.global_identity = global_identity
        self.children = [] #挑出来的类中心，才能作为树上的节点；否则都在cluster_member里面
        # self.cluster_member = [] 
        self.locations = []
        self.tour = []
    
    # 添加子节点
    def add_child(self, child):
        self.children.append(child)
    def  add_cluster_member(self,member):
        self.cluster_member.append(member)   


def print_tree(node, level=5):
    indent = '   |' * (level - 1) + '-|-' if level > 0 else ''
    print(indent + str(node.global_identity))
    for child in node.children:
        print_tree(child, level+1)


def test_node_tree():
    root = Node('root')
    node1 = Node('node11')
    node2 = Node('node12')
    root.add_child(node1)
    root.add_child(node2)
    print_tree(root)



def traverse_and_collect_global_identities(node):
    identities = [node.global_identity]
    for child in node.children:
        identities.extend(traverse_and_collect_global_identities(child))
    return identities





del_count = 0




import random




from sklearn.manifold import SpectralEmbedding
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA

def my_Graph_embeding_spectral(arg):
    misFilename,my_embeding_dim,MAX_NUM_NODES = arg
    G = nx.read_gpickle(misFilename)
    num_nodes = G.number_of_nodes()
    if num_nodes <1001:
        embedding = SpectralEmbedding(n_components=my_embeding_dim, affinity='precomputed')
        adjacency_matrix = nx.to_numpy_array(G)
        pos = embedding.fit_transform(adjacency_matrix)

    else:
        # adjacency_matrix = nx.to_scipy_sparse_matrix(G,dtype=np.float64)
        # values, pos = eigsh(adjacency_matrix, k=my_embeding_dim, which='SM',maxiter=5000)  # 'SM'表示求最小特征值
        adj_matrix = nx.adjacency_matrix(G).toarray()
        pca = PCA(n_components=my_embeding_dim)
        pos = pca.fit_transform(adj_matrix)
        pos = np.array([pos[i].tolist() for i in range(len(G.nodes()))])

    pos = pos.tolist()
    sub_num_nodes = 100   
    if len(pos) > sub_num_nodes:
        pos = random.sample(pos,sub_num_nodes)
    sub_num_nodes = len(pos)
    if misFilename[-11:-8] == "999":
        print(misFilename)
    pos = np.array(list(pos) + [np.ones(my_embeding_dim)*100])
    weis = np.array(list(ones(sub_num_nodes)*num_nodes/sub_num_nodes) + [MAX_NUM_NODES - num_nodes])
    return pos, weis 




import pickle

#  misFilename,point_num_threshold,ballRadius,kk

def my_RWD_coreset_spectral(global_misFilename_list,ballRadius_RWD=0.1,kk=4,maxPoolNum=96,my_embeding_dim=3,RWD_iter=5,MAX_NUM_NODES=1000):
    current_time0 = time.strftime("%H:%M:%S", time.localtime())
    print("current_time0 = ",current_time0)
    tmp_arg_list = [[fname,my_embeding_dim,MAX_NUM_NODES] for fname in global_misFilename_list]
    
    
    # tmp_res = []
    # for arg,cou in zip(tmp_arg_list,range(len(tmp_arg_list))):
    #     if cou %1000 == 0:
    #         print("------------------ cou = ",cou)
    #     res = my_Graph_embeding_spectral(arg)
    #     tmp_res.append(res)
        

    my_embeding_path = "my_dataCO/DIFCUSO_data/mis_er/embeded_pointset/spectral/"
    os.makedirs(my_embeding_path,exist_ok=True)
    my_embeding_path_file = my_embeding_path + "embeded_pointset" + str(len(tmp_arg_list))+ "_" + str(my_embeding_dim) + ".bin"
    if not(os.path.exists(my_embeding_path_file)):
        with Pool(maxPoolNum) as pool:
            tmp_res_0 = pool.map(my_Graph_embeding_spectral,tmp_arg_list[:125000])

        with Pool(maxPoolNum) as pool:
            tmp_res_1 = pool.map(my_Graph_embeding_spectral,tmp_arg_list[125000:])

        tmp_res = list(tmp_res_0) + list(tmp_res_1)  
        # np.save(my_embeding_path_file,tmp_res)
        with open(my_embeding_path_file, 'wb') as f:
            pickle.dump(tmp_res,f)
    else:
        # tmp_res = np.load(my_embeding_path_file)    
        with open(my_embeding_path_file, 'rb') as f:
            tmp_res = pickle.load(f)
    # with Pool(maxPoolNum) as pool:
    #     tmp_res = pool.map(my_Graph_embeding_spectral,tmp_arg_list)
    print("----------3000---------------------------",time.strftime("%H:%M:%S", time.localtime())  )
    
    
    # tmp_res = list(tmp_res1) + list(tmp_res2)
    locations_list = [tr[0] for tr in tmp_res]; weights_list = [tr[1] for tr in tmp_res]
    # del_list = [len(list(ll)) for ll in weights_list]
    # print(del_list)
    id_list_list = [np.arange(array(global_misFilename_list).shape[0])]
    coreset_id_list = []
    
    while id_list_list !=[]:  
        dist_list_list = [[] for i in range(len(id_list_list))]
        # print("")
        print("maxPoolNum = ",maxPoolNum)
        for k in range(kk):
            print("k = ",k)
            print("------- = ",time.strftime("%H:%M:%S", time.localtime()))
            size_list = [len(ll) for ll in id_list_list]        
            acc_size_list = [sum(size_list[:i+1]) for i in range(len(size_list))]
            if k==0:
                tmp_center_id_list = [random.choice(id_l) for id_l in id_list_list]
            coreset_id_list = coreset_id_list + tmp_center_id_list
            if id_list_list == []:
                id_list_list = [[0]]
            id_a_list = reduce(lambda x,y:list(x)+list(y),id_list_list)
            id_b_list = []
            for cl,s in zip(tmp_center_id_list,size_list):
                id_b_list = id_b_list + [cl]*s
            arg_list = [[locations_list[id_a],weights_list[id_a],locations_list[id_b],weights_list[id_b],RWD_iter] for id_a,id_b in zip(id_a_list,id_b_list)]        
            
            with Pool(maxPoolNum) as pool:
                tmp_res_list = pool.map(myMultiEmdRWD,arg_list) 
            tmp_dist_list = [r[2] for r in tmp_res_list]    
            tmp_dist_list_list = [list(np.array(tmp_dist_list)[ind0:ind1]) for ind0,ind1 in zip([0]+list(acc_size_list)[:-1], acc_size_list)]
            for i in range(len(tmp_dist_list_list)):
                dist_list_list[i] = list(dist_list_list[i])
                dist_list_list[i].append(tmp_dist_list_list[i])
            new_dist_list_list = [];new_id_list_list = []
            for id_l,dist_l in zip(id_list_list,dist_list_list):
                tmp_reserve_index_list = np.where(np.min(dist_l,axis=0) > ballRadius_RWD)[0]
                if array(tmp_reserve_index_list).shape[0] >0 :
                    new_id_list_list.append(id_l[tmp_reserve_index_list])
                    new_dist_list_list.append((array(dist_l)[:,tmp_reserve_index_list]))
            dist_list_list = new_dist_list_list; id_list_list = new_id_list_list
            tmp_center_index_list = [np.argmax(np.min(dist_l,axis=0)) for dist_l in dist_list_list]
            tmp_center_id_list = [id_l[i] for i,id_l in zip(tmp_center_index_list,new_id_list_list)]
            assert len(tmp_center_index_list)==len(new_id_list_list)
        new_id_list_list = []
        for k in range(kk):
            for id_l,dl in zip(id_list_list,dist_list_list):
                tmp_classify_index_list = np.argmax(dl,axis=0)
                tmp_reserve_index_list = array(np.where(tmp_classify_index_list==k)).flatten()
                if array(tmp_reserve_index_list).shape[0] > kk:
                    new_id_list_list.append(id_l[tmp_reserve_index_list])
                if array(tmp_reserve_index_list).shape[0] > 0 and array(tmp_reserve_index_list).shape[0] < kk+1:
                    coreset_id_list = list(coreset_id_list) + list(id_l[tmp_reserve_index_list])
                    # print("")
             
        id_list_list = new_id_list_list
        print("++++++++++++ = ",sum([len(list(id_l)) for id_l in id_list_list]),len(list(id_list_list)),len(coreset_id_list))                  
        current_time1 = time.strftime("%H:%M:%S", time.localtime())
        print("current_time0 = ",current_time0)
        print("current_time1 = ",current_time1)
    print("len(coreset_id_list) = ",len(coreset_id_list))        
    print(len(set(coreset_id_list)))
    return array(global_misFilename_list)[coreset_id_list]
   






def test_my_RWD_coreset():
    print("-"*50)
    global_misFilename_list = glob.glob("my_dataCO/DIFCUSO_data/mis_er/train_data_3000/*.gpickle")[125000:]
    my_RWD_coreset_spectral(global_misFilename_list,ballRadius_RWD=0.1,kk=4,maxPoolNum=40,my_embeding_dim=3,RWD_iter=5)
    print("")
    
    
    
# test_my_RWD_coreset()    