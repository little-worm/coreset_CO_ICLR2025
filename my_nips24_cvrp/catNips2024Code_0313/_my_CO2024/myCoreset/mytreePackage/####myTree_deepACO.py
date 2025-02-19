
import random,torch,os,sys,ot,time
import numpy as np
from multiprocessing import Pool
from numpy import array,arange
tmp_cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(tmp_cfd)
from functools import reduce





class Node:
    def __init__(self, global_identity):
        self.global_identity = global_identity
        self.children = [] #挑出来的类中心，才能作为树上的节点；否则都在cluster_member里面
        # self.cluster_member = [] 
        self.org_locations = []
        self.new_locations = []
        self.weights = []
    
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






def mypool_OT(arg):
    weis_a = arg[0]; weis_b = arg[1]; locs_a = arg[2]; locs_b = arg[3]
    M = ot.dist(locs_a,locs_b)
    return(ot.emd2(weis_a,weis_b,M))





def mypool_emdRWD_deepACO_cvrp(argList = 10):    
    weight_a,weight_b,location_a,location_b,maxIterTimes,my_assertErr = argList
    # centralized location_a
    # assert np.prod( np.abs(weight_a.dot(location_a) ) < my_assertErr), "'location_a' must have 0 mean"
    # assert np.prod( np.abs(weight_b.dot(location_b) ) < my_assertErr), "'location_b' must have 0 mean"
    # assert np.abs( sum(weight_a) - 1 ) < my_assertErr, "'sum(weight_a)==1' must hold !!!"
    # assert np.abs( sum(weight_b) - 1 ) < my_assertErr, "'sum(weight_b)==1' must hold !!!"
    # if sum(weight_a) < 1:
    #     index = np.argmax(weight_a)
    #     weight_a[index] += abs(1-sum(weight_a)) 
    # else:
    #     index = np.argmax(weight_a)
    #     weight_a[index] -= abs(1-sum(weight_a))
    # if sum(weight_b) < 1:
    #     index = np.argmax(weight_b)
    #     weight_b[index] += abs(1-sum(weight_b)) 
    # else:
    #     index = np.argmax(weight_b)
    #     weight_b[index] -= abs(1-sum(weight_b))    
    # ## 此处断言location_a,location_b是中心化过后的。
    # print("maxIterTimes for RWD = ",maxIterTimes)
    if len(location_a.shape)==3 and len(location_b.shape)==3:
        location_a = np.squeeze(location_a,axis=1)
        location_b = np.squeeze(location_b,axis=1)
    # print("location_a = ",location_a.shape,array(weight_a).shape)
    loss_pre = 10000000; loss_now = 10000000
    for iterTime in range(maxIterTimes):
        costMatrix = ot.dist(location_a,location_b)
        flowMartrix = ot.emd(weight_a, weight_b, costMatrix)
        loss_now = np.sum( np.array(costMatrix) * np.array(flowMartrix) )
    #    print("loss_now = ",loss_now)
        matrixB = (location_a.T).dot(flowMartrix).dot(location_b)
        matrixU,matrixS,matrixVT = np.linalg.svd(matrixB)
        diagList = list([1 for i in range(len(matrixB)-1)])
        diagList.append(np.linalg.det(matrixU)*np.linalg.det(matrixVT))
        matrixR = matrixU.dot( np.diag(  diagList  )).dot(matrixVT)
        location_b = location_b.dot(matrixR.T)
        if loss_pre - loss_now < my_assertErr:
            break
        else:
            loss_pre = loss_now
    #print("iterTime = ",iterTime)
    return flowMartrix, location_b, loss_now













def my_coreset_for_pointset(arg):
    global_locations_list,ballRadius,kk = arg[0],arg[1],arg[2]
    global_id_list = np.arange(len(list(global_locations_list)))
    id_list_list = [np.copy(global_id_list)]
    coreset_id_list = []; coreset_weights_list = []
    while len(id_list_list) > 0:
        new_id_list_list = []
        for id_list in id_list_list:
            dist_matrix = []
            for k in range(kk):
                if k==0:
                    center_id= random.choice(id_list)
                tmp_dist = ot.dist(array([global_locations_list[center_id]]),global_locations_list[id_list]).flatten()
                reserve_index_list = np.where(tmp_dist > ballRadius)[0]
                id_list = id_list[reserve_index_list]
                center_weight = len(np.where(tmp_dist < ballRadius)[0])   
                if center_weight>0:
                    coreset_id_list.append(center_id); coreset_weights_list.append(center_weight)

                dist_matrix.append(tmp_dist)
                dist_matrix = (array(dist_matrix).T[reserve_index_list]).T.tolist()
                if len(reserve_index_list)>0:
                    center_id_index = np.argmax(np.min(dist_matrix,axis=0))
                    center_id = id_list[center_id_index]
                # print("")
            classify_list = np.argmin(dist_matrix,axis=0)
            for k in range(kk):
                tmp_index_list = array(np.where(classify_list==k)).flatten()    
                
                if array(tmp_index_list).shape[0] > 0:
                    new_id_list_list.append(id_list[tmp_index_list])
        # print(len(coreset_id_list),sum([len(i) for i in new_id_list_list]),sum(coreset_weights_list),sum([len(i) for i in new_id_list_list])+sum(coreset_weights_list))        
        id_list_list = new_id_list_list   
    coreset_locationss_list = global_locations_list[coreset_id_list]
    coreset_weights_list = array(coreset_weights_list) / sum(coreset_weights_list)
    return coreset_locationss_list,coreset_weights_list  
        






 
   
   




def my_RWD_coreset_deepACO_cvrp(global_locations_list,global_weights_list,ballRadius_RWD=20,kk=4,maxPoolNum=40,ballRadius_pointset=1,point_num_threshold=100,maxIterTimes_RWD=1):
    current_time0 = time.strftime("%H:%M:%S", time.localtime())
    new_global_locations_list = np.copy(global_locations_list)
    # if array(global_locations_list).shape[1] > point_num_threshold:
    #     pass
    #     # tmp_arg_list = [[loc_l,ballRadius_pointset,kk] for loc_l in global_locations_list]
    #     # with Pool(maxPoolNum) as pool:
    #     #     tmp_res = pool.map(my_coreset_for_pointset,tmp_arg_list)
    #     # small_global_locations_list = [tr[0] for tr in tmp_res]; small_global_weights_list = [tr[1] for tr in tmp_res]
    # else:
    small_global_locations_list = np.copy(global_locations_list) 
    small_global_weights_list = global_weights_list
    
    id_list_list = [np.arange(array(global_locations_list).shape[0])]
    coreset_id_list = []
    
    my_root = Node('root')
    father_node_list = [my_root]
    while id_list_list !=[]:  
        tmp_node_list_list = [[] for i in range(len(id_list_list))]  
        dist_list_list = [[] for i in range(len(id_list_list))]
        # print("")
        for k in range(kk):
            size_list = [len(ll) for ll in id_list_list]        
            acc_size_list = [sum(size_list[:i+1]) for i in range(len(size_list))]
            if k==0:
                tmp_center_id_list = [random.choice(id_l) for id_l in id_list_list]
            assert len(father_node_list) == len(tmp_center_id_list) 
            # assert len(father_node_list) == len(id_list_list)
            for i in range(len(tmp_center_id_list)):
                # add child
                tmp_node_id = tmp_center_id_list[i]
                tmp_node = Node(tmp_node_id); tmp_node.org_locations = global_locations_list[tmp_node_id]; tmp_node.new_locations = new_global_locations_list[tmp_node_id]; tmp_node.weights = global_weights_list[tmp_node_id]
                father_node_list[i].add_child(tmp_node)
                tmp_node_list_list[i] = list(tmp_node_list_list[i])
                tmp_node_list_list[i].append(tmp_node)
            coreset_id_list = coreset_id_list + tmp_center_id_list
            id_b_list = reduce(lambda x,y:list(x)+list(y),id_list_list)
            id_a_list = []
            for cl,s in zip(tmp_center_id_list,size_list):
                id_a_list = id_a_list + [cl]*s
            arg_list = [[small_global_weights_list[id_a],small_global_weights_list[id_b],new_global_locations_list[id_a],new_global_locations_list[id_b],maxIterTimes_RWD,0.00001] for id_a,id_b in zip(id_a_list,id_b_list)]        
            with Pool(maxPoolNum) as pool:
                tmp_res_list = pool.map(mypool_emdRWD_deepACO_cvrp,arg_list) 
            tmp_dist_list = [r[2] for r in tmp_res_list]    
            tmp_dist_list_list = [list(np.array(tmp_dist_list)[ind0:ind1]) for ind0,ind1 in zip([0]+list(acc_size_list)[:-1], acc_size_list)]
            tmp_locs_list = tmp_dist_list = [r[1] for r in tmp_res_list]
            new_global_locations_list[id_b_list] = tmp_locs_list
            for i in range(len(tmp_dist_list_list)):
                dist_list_list[i] = list(dist_list_list[i])
                dist_list_list[i].append(tmp_dist_list_list[i])
            new_dist_list_list = [];new_id_list_list = []; new_tmp_node_list_list = []; new_father_node_list = []
            for id_l,dist_l,tmp_node_l,f_node in zip(id_list_list,dist_list_list,tmp_node_list_list,father_node_list):
                tmp_reserve_index_list = np.where(np.min(dist_l,axis=0) > ballRadius_RWD)[0]
                # print("tmp_reserve_index_list = ",array(tmp_reserve_index_list).shape,len(dist_l[0]))
                if array(tmp_reserve_index_list).shape[0] >0 :
                    new_id_list_list.append(id_l[tmp_reserve_index_list])
                    new_dist_list_list.append((array(dist_l)[:,tmp_reserve_index_list]))
                    new_tmp_node_list_list.append(tmp_node_l)
                    new_father_node_list.append(f_node)
            dist_list_list = new_dist_list_list; id_list_list = new_id_list_list; tmp_node_list_list = new_tmp_node_list_list; father_node_list = new_father_node_list
            tmp_center_index_list = [np.argmax(np.min(dist_l,axis=0)) for dist_l in dist_list_list]
            tmp_center_id_list = [id_l[i] for i,id_l in zip(tmp_center_index_list,new_id_list_list)]
            assert len(tmp_center_index_list)==len(new_id_list_list)
        new_id_list_list = []; new_tmp_node_list_list = []
        for k in range(kk):
            # print("---------------- k = ",k)
            for id_l,dl,tmp_node_l in zip(id_list_list,dist_list_list,tmp_node_list_list):
                tmp_classify_index_list = np.argmax(dl,axis=0)
                tmp_reserve_index_list = array(np.where(tmp_classify_index_list==k)).flatten()
                # print("===================== = ",array(tmp_reserve_index_list).shape)
                if array(tmp_reserve_index_list).shape[0] > kk:
                    # print("+++++++++++++++++++++++++++= ")
                    new_id_list_list.append(id_l[tmp_reserve_index_list])
                    new_tmp_node_list_list.append(tmp_node_l[k])
                if array(tmp_reserve_index_list).shape[0] < kk+1 and len(tmp_reserve_index_list) > 0:
                    # [tmp_node_l[k].add_child(Node(t_id)) for t_id in id_l[tmp_reserve_index_list]]
                    for t_id in id_l[tmp_reserve_index_list]:
                        tmp_node = Node(t_id)
                        tmp_node.org_locations = global_locations_list[t_id]; tmp_node.new_locations = new_global_locations_list[t_id]
                        tmp_node.weights = global_weights_list[t_id]
                        tmp_node_l[k].add_child(tmp_node)
                    coreset_id_list = list(coreset_id_list) + list(id_l[tmp_reserve_index_list])
             
        id_list_list = new_id_list_list; father_node_list =  np.copy(new_tmp_node_list_list) 
        print("++++++++++++ = ",sum([len(list(id_l)) for id_l in id_list_list]),len(list(coreset_id_list)))                  
        current_time1 = time.strftime("%H:%M:%S", time.localtime())
        print("current_time0 = ",current_time0)
        print("current_time1 = ",current_time1)
    print("len(coreset_id_list) = ",len(coreset_id_list))        
    collect_id = traverse_and_collect_global_identities(my_root)    
    print("collect_id = ",len(collect_id))
    print(len(set(coreset_id_list)),len(set(collect_id)),set(coreset_id_list) < set(collect_id))
    return array(new_global_locations_list)[coreset_id_list],coreset_id_list,my_root
   



def test_my_RWD_coreset_deepACO_cvrp():
    print("-"*50)
    print("-"*50)
    n = 100; m = 100
    global_locations_list = np.random.rand(n,m,2)
    global_weights_list = np.random.rand(n,m) 
    global_weights_list = [wei/sum(wei) for wei in global_weights_list]
    global_locations_list = [locs-wei.dot(locs) for wei,locs in zip(global_weights_list,global_locations_list)]

    my_RWD_coreset_deepACO_cvrp(global_locations_list,global_weights_list,ballRadius_RWD=0.01,kk=4,maxPoolNum=40,ballRadius_pointset=1,point_num_threshold=100,maxIterTimes_RWD=1)


# test_my_RWD_coreset_deepACO_cvrp()





