
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
from catNips2024Code_0313._my_CO2024.myCoreset_mis.mytreePackage.myRWD import myMultiEmdRWD
import umap
from node2vec import Node2Vec
import pickle
from scipy.sparse import csgraph

# import cudf
# import cuml
# from cuml.manifold import MDS as cuMDS

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










# def myPoolEmdGWD(arg):
#     C_a, C_b, weight_a, weight_b = arg[0],arg[1],arg[2],arg[3]
#     # print(array(weight_a).shape[0],array(weight_b).shape[0])
#     if np.array_equal(C_a,C_b) and np.array_equal(weight_a,weight_b):
#         # print("res = 0")
#         return 0
#     # res = ot.gromov.entropic_gromov_wasserstein2(C_a,C_b,weight_a,weight_b)
#     res = ot.gromov.gromov_wasserstein2(C_a,C_b,weight_a,weight_b)
#     return res




del_count = 0




# def my_coreset_for_pointset_mis(arg):
#     misFilename,point_num_threshold,ballRadius,kk = arg[0],arg[1],arg[2],arg[3]
#     G = nx.read_gpickle(misFilename)
#     global_node_list = np.array(G.nodes)
#     node_subset,node_weights = np.copy(global_node_list), ones(global_node_list.shape[0]) / global_node_list.shape[0]
#     subgraph = G.subgraph(node_subset)
#     shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
#     node_list = list(shortest_path_lengths.keys())
#     num_nodes = len(node_list)
#     shortest_path_matrix = np.zeros((num_nodes, num_nodes))
#     for i, node1 in enumerate(node_list):
#         for j, node2 in enumerate(node_list):
#             shortest_path_matrix[i, j] = shortest_path_lengths[node1].get(node2, float('inf'))
    
#     if global_node_list.shape[0] < point_num_threshold + 1:
#         return shortest_path_matrix, ones(num_nodes) / num_nodes
#     global_id_list = np.arange(len(list(global_node_list)))
#     id_list_list = [np.copy(global_id_list)]
#     coreset_id_list = []; coreset_weights_list = []
#     while len(id_list_list) > 0:
#         new_id_list_list = []
#         for id_list in id_list_list:
#             dist_matrix = []
#             for k in range(kk):
#                 if k==0:
#                     center_id= random.choice(id_list)
#                 tmp_dist = shortest_path_matrix[center_id]; tmp_dist = array(tmp_dist)[id_list]
#                 reserve_index_list = np.where(tmp_dist > ballRadius)[0]
#                 id_list = id_list[reserve_index_list]
#                 center_weight = len(np.where(tmp_dist < ballRadius)[0])   
#                 if center_weight>0:
#                     coreset_id_list.append(center_id); coreset_weights_list.append(center_weight)

#                 dist_matrix.append(tmp_dist)
#                 dist_matrix = (array(dist_matrix).T[reserve_index_list]).T.tolist()
#                 if len(reserve_index_list)>0:
#                     center_id_index = np.argmax(np.min(dist_matrix,axis=0))
#                     center_id = id_list[center_id_index]
#                 # print("")
#             classify_list = np.argmin(dist_matrix,axis=0)
#             for k in range(kk):
#                 tmp_index_list = array(np.where(classify_list==k)).flatten()    
                
#                 if array(tmp_index_list).shape[0] > 0:
#                     new_id_list_list.append(id_list[tmp_index_list])
#         id_list_list = new_id_list_list   
#     coreset_shortest_path_matrix = shortest_path_matrix[coreset_id_list].T[coreset_id_list]
#     coreset_shortest_path_matrix = coreset_shortest_path_matrix * num_nodes /100
#     coreset_weights_list = array(coreset_weights_list) / sum(coreset_weights_list)
#     # print(coreset_shortest_path_matrix.max(),num_nodes)
#     global del_count
#     del_count = del_count + 1
#     # if (del_count%32) == 0:
#     #     print(" = ",arg[0])
#     #     print("graph size = ",coreset_weights_list.shape[0])
#     return coreset_shortest_path_matrix,coreset_weights_list  
        







# def test_my_coreset_for_pointset_mis():
#     global_misFilename_list = glob.glob("my_dataCO/DIFCUSO_data/mis_er/test_data/er-90-100/*.gpickle")
#     ballRadius_forPointset = 1
#     arg = [global_misFilename_list[0],ballRadius_forPointset,100,4]
#     my_coreset_for_pointset_mis(arg)

# test_my_coreset_for_pointset_mis()







def my_Graph_embeding(arg):
    misFilename,my_embeding_dim,maxPoolNum = arg
    # G = nx.read_gpickle(misFilename)
    G = nx.read_gpickle(misFilename)
    node_list = list(G.nodes())
    if len(node_list) > 100: 
        nodes_of_interest = random.sample(node_list,100)
    else:
        nodes_of_interest = node_list
    subgraph = G.subgraph(nodes_of_interest)
    sub_num_nodes = subgraph.number_of_nodes()
    if sub_num_nodes < 100000:
        shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
        node_list = list(shortest_path_lengths.keys())
        shortest_path_matrix = np.zeros((sub_num_nodes, sub_num_nodes))
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list):
                # t_del = shortest_path_lengths[node1].get(node2, float('inf'))
                # print("!!!!!!!!!!!!!!!!!!!!!! = ",t_del)
                shortest_path_matrix[i, j] = shortest_path_lengths[node1].get(node2, float('inf'))
        shortest_path_matrix = np.nan_to_num(shortest_path_matrix,nan=100) * (sub_num_nodes / 100)**2
        # if sub_num_nodes >100:
        #     select_indexs = random.sample(arange(sub_num_nodes),100)
        #     shortest_path_matrix = shortest_path_matrix[select_indexs,select_indexs]
        # print("np.isnan(shortest_path_matrix).sum() = ",np.isnan(shortest_path_matrix).sum())
        mds = MDS(n_components=my_embeding_dim, dissimilarity='precomputed', random_state=42,normalized_stress='auto',n_jobs=maxPoolNum)
        pos = mds.fit_transform(shortest_path_matrix)
        # 将numpy数组转换为cudf数据框
        # dist_matrix_cudf = cudf.DataFrame.from_records(shortest_path_matrix)
        # # 使用cuML进行MDS
        # mds_gpu = cuMDS(n_components=my_embeding_dim, dissimilarity='precomputed')
        # pos = mds_gpu.fit_transform(dist_matrix_cudf)
    if sub_num_nodes > 10000000:
        # embedding = SpectralEmbedding(n_components=my_embeding_dim,n_jobs=maxPoolNum)
        # pos = embedding.fit_transform(nx.to_numpy_array(subgraph))
        
        #---------------------------
        adj_matrix = nx.adjacency_matrix(subgraph).todense()
        # 转换为稀疏矩阵
        adj_matrix_sparse = csgraph.csgraph_from_dense(adj_matrix, null_value=0)
        # 使用 SpectralEmbedding 进行降维
        pos = SpectralEmbedding(n_components=my_embeding_dim, affinity='precomputed',n_jobs=maxPoolNum).fit_transform(adj_matrix_sparse)

        #------------------
        # adj_matrix = nx.to_scipy_sparse_matrix(G)
        # reducer = umap.UMAP(n_components=my_embeding_dim)
        # pos = reducer.fit_transform(adj_matrix)
        #--------------------------------
        # 初始化 Node2Vec 对象，设置参数
        # node2vec = Node2Vec(G, dimensions=my_embeding_dim, walk_length=10, num_walks=1, workers=32)
        # model = node2vec.fit(window=10, min_count=1, batch_words=4)
        # pos = np.array([model.wv[str(node)] for node in G.nodes()])


    # del_count = del_count + 1
    if misFilename[-11:-8] == "000":
        print(misFilename)
    # print("sub_num_nodes ==================== ",sub_num_nodes)
    pos = np.array(pos)
    pos = pos - pos.mean(axis=0)
    return pos, ones(sub_num_nodes) / sub_num_nodes








#  misFilename,point_num_threshold,ballRadius,kk

def my_RWD_coreset(global_misFilename_list,ballRadius_RWD=0.1,kk=4,maxPoolNum=96,my_embeding_dim=3,RWD_iter=5):
    current_time0 = time.strftime("%H:%M:%S", time.localtime())
    print("current_time0 = ",current_time0)
    tmp_arg_list = [[fname,my_embeding_dim,1] for fname in global_misFilename_list]
    # tmp_res2 = []
    # print("maxPoolNum = ",maxPoolNum)
    # with Pool(maxPoolNum) as pool:
    #     tmp_res1 = pool.map(my_Graph_embeding,tmp_arg_list[:-3000])
    # print("----------3000---------------------------",time.strftime("%H:%M:%S", time.localtime())  )
    # for arg in tmp_arg_list[-3000:]:
    #     t_res = my_Graph_embeding(arg)
    #     tmp_res2.append(t_res)
    # print("----------3000---------------------------",time.strftime("%H:%M:%S", time.localtime())  )
    
    
    # tmp_res = []
    # for arg in tmp_arg_list:
    #     t_res = my_Graph_embeding(arg)
    #     tmp_res.append(t_res)
    # print("----------3000---------------------------",time.strftime("%H:%M:%S", time.localtime())  )
    
    
    with Pool(maxPoolNum) as pool:
        tmp_res = pool.map(my_Graph_embeding,tmp_arg_list)
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
    global_misFilename_list = glob.glob("my_dataCO/DIFCUSO_data/mis_er/train_data_3000/*.gpickle")[6000:7000]
    my_RWD_coreset(global_misFilename_list,ballRadius_RWD=0.1,kk=4,maxPoolNum=40,my_embeding_dim=3,RWD_iter=5)
    print("")
    
    
    
# test_my_RWD_coreset()    