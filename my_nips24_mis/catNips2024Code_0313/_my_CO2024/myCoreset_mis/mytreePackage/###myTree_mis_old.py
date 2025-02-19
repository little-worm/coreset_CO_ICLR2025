import random,torch,os,sys,ot
import numpy as np
from multiprocessing import Pool
import glob
tmp_cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(tmp_cfd)

from catNips2024Code_0313._my_CO2024.myCoreset_mis.mytreePackage.myGWD_old import myMultiEmdGWD
from functools import reduce
import time,pickle
import networkx as nx
from networkx.algorithms import shortest_paths

# 定义节点类
class Node:
    def __init__(self, global_identity):
        self.global_identity = global_identity
        self.children = [] #挑出来的类中心，才能作为树上的节点；否则都在cluster_member里面
        self.cluster_member = [] 
        self.pickleFilename = []
        #self.solutionFilename = []
    
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




#   test_node_tree()






def my_reshapeList(org_list,example_list):
    tmp_index = [len(li) for li in example_list]      
    org_lists = []
    ind00 = 0
    for ind in tmp_index:
        ind11 = ind00 + ind
        org_lists.append( org_list[ind00:ind11] )
        ind00 = ind11
    return org_lists    

def test_my_reshapeList():
    org_list = [i for i in range(10)]
    example_list = [[0],[1,1,1],[2,2,2],[3,3,3]]
    res = my_reshapeList(org_list,example_list)
    print(res)
    
#   test_my_reshapeList()    







def my_flatenList(ll):
    res = []
    for l in ll:
        res = res + l
    return res

def test_my_flatenList():
    ll = [[1,2,3],[4],[5,6],[7,89,0]]
    res = my_flatenList(ll)
    print(res)
    
#   test_flatenList() 






random.seed(1234); maxIterTimes = 10; my_assertErr = 0.0000001
np.random.seed(1234)




#def myTreeCoreset(global_costmatrixFilename_list,ballRadius=20,kk = 4,maxPoolNum=10,Filename='tree.pkl'):
#    print("ballRadius,kk,maxPoolNum = ",ballRadius,kk,maxPoolNum)
#    global_costmatrix_list = []
#    for cf in global_costmatrixFilename_list:
#        tmp = np.load(cf)
#        global_costmatrix_list.append(tmp)
#        print("-----------------")
#    print("==================")
    
    
    
    
def myTreeCoreset(global_misFilename_list,ballRadius=20,kk = 4,maxPoolNum=10,Filename='tree.pkl'):
    print("ballRadius,kk,maxPoolNum = ",ballRadius,kk,maxPoolNum)
    #debug_global_locations_list_0 = np.copy(global_picklefile_list)
    global_costmatrix_list = []
    for mis_fn in global_misFilename_list:
        tmp_G = nx.read_gpickle(mis_fn)
        tmp_lengths_generator = shortest_paths.all_pairs_shortest_path_length(tmp_G)
    
        # 将生成器对象转换为字典
        tmp_lengths_generator = dict(tmp_lengths_generator)
        # 提取节点列表
        tmp_nodes = list(tmp_G.nodes())
        # 初始化一个距离矩阵
        tmp_costmatrix = np.zeros((len(tmp_nodes), len(tmp_nodes)))
        # 填充距离矩阵
        for i, node_i in enumerate(tmp_nodes):
            for j, node_j in enumerate(tmp_nodes):
                if node_j in tmp_lengths_generator[node_i]:
                    tmp_costmatrix[i, j] = tmp_lengths_generator[node_i][node_j]
                else:
                    tmp_costmatrix[i, j] = np.inf  # 如果没有路径则设置为无穷大

        global_costmatrix_list.append(tmp_costmatrix)
    identity_list = [i for i in range(len(global_costmatrix_list))]
    print("111111111111111")
    new_identity_lists = [identity_list]
    root = Node('root') # -1 means root node
    root_list = [root]
    debug_cluster_lists = []
    child_id_flatenList = []
    child_weight_flatenList = []
    while new_identity_lists != [] :
        print("-"*50)
        current_time = time.strftime("%H:%M:%S", time.localtime())    
        print("current_time = ",current_time)
        child_id_lists = []
        identity_lists = new_identity_lists
        min_dist_lists = [ list(np.random.rand(len(ll)) + 100000000) for ll in identity_lists ]
        child_lists = [[] for i in range(len(identity_lists))]
        child_id_lists = [[] for i in range(len(identity_lists))]
        kk_dist_lists = []
        for i in range(kk):
            arg_id_a_list = []
            arg_id_b_list = []
            for id_list,md_list,ro,ii in zip(identity_lists,min_dist_lists,root_list,range(len(identity_lists))):
                child_id = id_list[min(np.argmax(md_list),len(id_list))]
                if not(child_id in child_id_lists[ii]):
                    child_id_lists[ii].append(child_id)
                    child_node = Node(child_id)
                    child_lists[ii].append(child_node)
#                    root.add_child(child_node)
                else:
                    child_lists[ii].append(None)
                tmpArg_id_a_list = [ child_id for i in range(len(id_list)) ]
                tmpArg_id_b_list = id_list
                arg_id_a_list = arg_id_a_list +  tmpArg_id_a_list
                arg_id_b_list = arg_id_b_list + tmpArg_id_b_list
            argList = [[global_costmatrix_list[a_id],global_costmatrix_list[b_id]] for a_id,b_id in zip(arg_id_a_list,arg_id_b_list) ]
            len_argList = len(argList)
            print("len_argList = ",len_argList)

            if len_argList > 16:
                myPoolNmu = 32
            elif len_argList < 16 and len_argList > 8:
                myPoolNmu = 16
            elif len_argList < 8 and len_argList > 4:
                myPoolNmu = 8
            elif len_argList < 4 and len_argList > 2:
                myPoolNmu = 4
            else:
                myPoolNmu = len_argList

            with Pool(myPoolNmu) as pool:
                tmp_res = pool.map(myMultiEmdGWD,argList) 
            #for res,b_id in zip(tmp_res,arg_id_b_list): 
            #    global_locations_list[b_id] = res[1]
            tmp_dist_list = [res for res in tmp_res]
            kk_dist_lists.append(tmp_dist_list)
            min_dist_list = np.min(kk_dist_lists,axis=0)  
            min_dist_lists = my_reshapeList(min_dist_list,identity_lists)
        for ro,chs in zip(root_list,child_lists):
            for ch in chs:
                if ch!=None:
                    ro.add_child(ch)
                    ch.pickleFilename = global_misFilename_list[ch.global_identity]
                    #ch.solutionFilename = global_solutionFilename_list[ch.global_identity]

        child_flatenList = my_flatenList(child_lists)
        print(" ")
#        root_list = [ch for ch in child_flatenList if ch!=None]
        labelIndex_list =  np.argmin(kk_dist_lists,axis=0)
        
        labelIndex_lists = my_reshapeList(labelIndex_list,identity_lists)
        mask_labelIndex_lists = [ list( np.ones(len(ll),dtype=int) * (len(cl)-1) ) for ll,cl in zip(labelIndex_lists,child_id_lists) ]
        mask_labelIndex_list = my_flatenList(mask_labelIndex_lists)
        labelIndex_list = np.min([labelIndex_list,mask_labelIndex_list],axis=0)
        labelIndex_lists = my_reshapeList(labelIndex_list,identity_lists)
        new_identity_lists = []
        treeFlag_identity_lists = []
        for l_list,md_list,id_list,c_list in zip(labelIndex_lists,min_dist_lists,identity_lists,child_lists):
            tmp_leftGid_lists = [[] for i in range(kk)] 
            tmp_debug_cluster = [[] for i in range(kk)] 
            for l,d,id in zip(l_list,md_list,id_list):
                if d < ballRadius:
                    c_list[l].add_cluster_member( id )
                    tmp_debug_cluster[l].append(id)         
                    #add_clu
                else:
                    tmp_leftGid_lists[l].append( id )
            for ll in tmp_leftGid_lists:
                if ll != []:
                    new_identity_lists.append(ll) 
                    treeFlag_identity_lists.append(1)
                else:
                    treeFlag_identity_lists.append(0)
            tmp_debug_cluster = [ ll for ll in tmp_debug_cluster if ll!=[] ]
            debug_cluster_lists = debug_cluster_lists + tmp_debug_cluster 
        root_list = [ch for ch,flag in zip(child_flatenList,treeFlag_identity_lists) if flag==1]
        child_id_flatenList = child_id_flatenList + my_flatenList(child_id_lists)
        tmp_child_weight_flatenList = [len(ch.cluster_member) for ch in child_flatenList if ch!=None]
        child_weight_flatenList = child_weight_flatenList + tmp_child_weight_flatenList
        #print("debug_cluster = ",debug_cluster)
        #print("new_identity_lists = ",new_identity_lists)
        #print("child_id_lists = ",child_id_lists)    
    #print("debug_cluster_lists = ",debug_cluster_lists)
    #print("global_locations_list = ",global_locations_list)
    #print("child_id_flatenList = ",child_id_flatenList,len(child_id_flatenList))    
    print("child_weight_flatenList = ",sum(child_weight_flatenList))
    #print_tree(root,0)
    Filename = Filename + '_n' + '_r' + str(ballRadius) + '_kk' + str(kk)+ '_pool' + str(maxPoolNum) + '_' + str(len(child_id_flatenList))

    return child_id_flatenList,child_weight_flatenList,root
    



def test_myTreeCoreset():
    current_time0 = time.strftime("%H:%M:%S", time.localtime())    
    global_misFilename_list =  glob.glob("/root/autodl-tmp/cat_nips24_mis_0728/my_dataCO/DIFCUSO_data/mis/er_test/*.gpickle")
    global_misFilename_list = global_misFilename_list[:3]
    myTreeCoreset(global_misFilename_list,ballRadius=20,kk = 4,maxPoolNum=10,Filename='tree.pkl')
    current_time1 = time.strftime("%H:%M:%S", time.localtime())
    print("current_time0 = ",current_time0)
    print("current_time1 = ",current_time1)
    print("-"*50)



# test_myTreeCoreset()











