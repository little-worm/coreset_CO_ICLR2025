import random,os,sys,pickle,time
from multiprocessing import Pool
#tmp_cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append("./")
from my_global import *
import numpy as np
from numpy import ones
from mytreePackage.myReadWriteTSP import my_read_tspData,my_save_tspData
# from mytreePackage.myTree import myTreeCoreset
from mytreePackage.myRWD_old import myEmdRWD




def my_align_single_testData(args):
    test_single_dataLocations,test_single_dataWeights,myTreeFilename,top_num,max_iter,tree_mode = args
    with open(myTreeFilename, 'rb') as file:
        tmp_node = pickle.load(file)
    if tree_mode == 'org':
        top_num_dataLocations = [tmp_node.children[0].org_locations for i in range(top_num)]
    if tree_mode == 'new':
        top_num_dataLocations = [tmp_node.children[0].new_locations for i in range(top_num)]
    top_num_cost = [1000000 for i in range(top_num)]
    # print('----------22222-----------------')
    
    while len(tmp_node.children)>0:
        #print('=============================111111=====')
        tmp_child_list = tmp_node.children
        layer_cost_list = []
        for t_child in tmp_child_list:
            recorded_dist = max(top_num_cost)
            t_id = t_child.global_identity
            if tree_mode == 'new':
                t_locations = t_child.new_locations
            if tree_mode == 'org':
                t_locations = t_child.org_locations
            t_weights = np.ones(len(t_locations)) / len(t_locations)
            _, t_location, t_loss = myEmdRWD(t_locations,t_weights,test_single_dataLocations,test_single_dataWeights,maxIterTimes=max_iter)
            layer_cost_list.append(t_loss)
            if t_loss<max(top_num_cost):  
                tmp_index = top_num_cost.index(recorded_dist)
                top_num_dataLocations[tmp_index] = t_location
                top_num_cost[tmp_index] = t_loss
                #print('t_loss, top_num_cost = ',t_loss,top_num_cost)
        tmp_node = tmp_child_list[layer_cost_list.index(min(layer_cost_list))] 
    #print('top_num_cost ---------------------------- = ',top_num_cost)
    return top_num_dataLocations
    





def test_my_align_single_testData():

    #orgTestdata_filename = 'test_TSP200_n128'
    n = 1280; m = 100; 
    org_test_tspFilename = 'my_dataCO/DIFCUSO_data/tsp100_3d/tsp100_test_Uniform_seed666_1280_3d.txt'
    org_test_location_lists,_ = my_read_tspData(org_test_tspFilename,n)
    org_test_location_lists = [ll-np.mean(ll,axis=0) for ll in org_test_location_lists]
    org_weights_lists = [np.ones(m)/m for i in range (n)]

    single_data_index = np.random.randint(0,n)
    test_single_dataLocations = org_test_location_lists[single_data_index]
    test_single_dataWeights = org_weights_lists[single_data_index]
    myTreeFilename = "my_dataCO/DIFCUSO_data/tsp100_3d/myRWD_data/res_tsp100_train_Guass_0_0.5_seed1234_1280000_3d_r0.028_kk4_pool32_5382/tree.plk"
    top_num = 1
    max_iter = 5
    args = [test_single_dataLocations,test_single_dataWeights,myTreeFilename,top_num,max_iter]
    my_align_single_testData(args)




#test_my_align_single_testData()






def my_align_all_testData(myTreeFilename,org_test_tspFilename,N,top_num,maxPoolNum,point_dim,max_iter = 5,tree_mode = 'new'):
    org_test_location_lists,org_test_tour_lists = my_read_tspData(org_test_tspFilename,N)
    org_test_location_lists = [ll-np.mean(ll,axis=0) for ll in org_test_location_lists]
    org_weights_lists = [ones(len(list(loc))) / len(list(loc)) for loc in org_test_location_lists]
    
    args_list = [[t_loc,t_wei,myTreeFilename,top_num,max_iter,tree_mode] for t_loc,t_wei in zip(org_test_location_lists,org_weights_lists)]
    with Pool(maxPoolNum) as pool:
        tmp_res = pool.map(my_align_single_testData,args_list) 
    flatten_data_list = []; flatten_tour_list = []
    for data_s,tour in zip(tmp_res,org_test_tour_lists):
        for data in data_s:
            flatten_data_list.append(data)
            flatten_tour_list.append(tour)
    save_file =  os.path.split(myTreeFilename)[0] + '/' + os.path.split(org_test_tspFilename)[1][:-4] + '_aligned.txt'
    print('save_file = ',save_file)
    my_save_tspData(save_file,flatten_data_list,flatten_tour_list,point_dim)



current_time0 = time.strftime("%H:%M:%S", time.localtime())    
N = 128
maxPoolNum = 40
top_num = 1
point_dim = 2; dim_mode = "d"
point_dim = 3; dim_mode = "ddim"

org_test_tspFilename_list = [ 
                            #  "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/tsp100_test_Guass_0_1_seed666_1280_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/tsp100_test_Guass_0_2_seed666_1280_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/tsp100_test_Guass_0_4_seed666_1280_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/tsp100_test_Guass_0_8_seed666_1280_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/tsp100_test_Uniform_seed666_1280_2d.txt", 
                              #------------
                            #  "my_dataCO/DIFCUSO_data/tsp200_2d_plus3000/tsp200_test_Guass_0_1_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp200_2d_plus3000/tsp200_test_Guass_0_2_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp200_2d_plus3000/tsp200_test_Guass_0_4_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp200_2d_plus3000/tsp200_test_Guass_0_8_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp200_2d_plus3000/tsp200_test_Uniform_seed666_128_2d.txt",
                            #   #------------
                            #  "my_dataCO/DIFCUSO_data/tsp500_2d_plus3000/tsp500_test_Guass_0_1_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp500_2d_plus3000/tsp500_test_Guass_0_2_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp500_2d_plus3000/tsp500_test_Guass_0_4_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp500_2d_plus3000/tsp500_test_Guass_0_8_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp500_2d_plus3000/tsp500_test_Uniform_seed666_128_2d.txt", 
                            #   #------------
                            #  "my_dataCO/DIFCUSO_data/tsp1000_2d_plus3000/tsp1000_test_Guass_0_1_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp1000_2d_plus3000/tsp1000_test_Guass_0_2_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp1000_2d_plus3000/tsp1000_test_Guass_0_4_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp1000_2d_plus3000/tsp1000_test_Guass_0_8_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp1000_2d_plus3000/tsp1000_test_Uniform_seed666_128_2d.txt", 
                              #------------
                            #  "my_dataCO/DIFCUSO_data/tsp10000_2d_plus3000/tsp10000_test_Guass_0_1_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp10000_2d_plus3000/tsp10000_test_Guass_0_2_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp10000_2d_plus3000/tsp10000_test_Guass_0_4_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp10000_2d_plus3000/tsp10000_test_Guass_0_8_seed666_128_2d.txt",
                            #  "my_dataCO/DIFCUSO_data/tsp10000_2d_plus3000/tsp10000_test_Uniform_seed666_128_2d.txt",
                            #---------------
                            # "my_dataCO/DIFCUSO_data/tsplib/my_tsblib_Eud2d_threshod100-1000.txt"
                            #   #--------------------------------------------------------------------------------
                            #   "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/tsp100_test_Guass_0_1_seed666_1280_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/tsp100_test_Guass_0_2_seed666_1280_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/tsp100_test_Guass_0_4_seed666_1280_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/tsp100_test_Guass_0_8_seed666_1280_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/tsp100_test_Uniform_seed666_1280_3ddim.txt",
                            #   #------------
                            #   "my_dataCO/DIFCUSO_data/tsp200_3ddim_plus3000/tsp200_test_Guass_0_1_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp200_3ddim_plus3000/tsp200_test_Guass_0_2_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp200_3ddim_plus3000/tsp200_test_Guass_0_4_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp200_3ddim_plus3000/tsp200_test_Guass_0_8_seed666_128_3ddim.txt",
                              "my_dataCO/DIFCUSO_data/tsp200_3ddim_plus3000/tsp200_test_Uniform_seed666_128_3ddim.txt", 
                            #   #------------
                            #   "my_dataCO/DIFCUSO_data/tsp500_3ddim_plus3000/tsp500_test_Guass_0_1_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp500_3ddim_plus3000/tsp500_test_Guass_0_2_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp500_3ddim_plus3000/tsp500_test_Guass_0_4_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp500_3ddim_plus3000/tsp500_test_Guass_0_8_seed666_128_3ddim.txt",
                              "my_dataCO/DIFCUSO_data/tsp500_3ddim_plus3000/tsp500_test_Uniform_seed666_128_3ddim.txt",
                            #   #------------
                            #   "my_dataCO/DIFCUSO_data/tsp1000_3ddim_plus3000/tsp1000_test_Guass_0_1_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp1000_3ddim_plus3000/tsp1000_test_Guass_0_2_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp1000_3ddim_plus3000/tsp1000_test_Guass_0_4_seed666_128_3ddim.txt",
                            #   "my_dataCO/DIFCUSO_data/tsp1000_3ddim_plus3000/tsp1000_test_Guass_0_8_seed666_128_3ddim.txt",
                              "my_dataCO/DIFCUSO_data/tsp1000_3ddim_plus3000/tsp1000_test_Uniform_seed666_128_3ddim.txt" 
                              #------------
                            ]


myTreeFilename_list = [
                        # "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/trainN01_myRWD_data/res_tsp100_train_Guass_0_1_seed1234_128000_2dplus_3000_r0.065_kk4_pool40IterRWD5_4003/tree.plk",
                        # "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/trainN01_myRWD_data/res_tsp100_train_Guass_0_1_seed1234_128000_2dplus_3000_r0.054_kk4_pool40IterRWD5_8245/tree.plk",
                        # "/my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/trainN01_myRWD_data/res_tsp100_train_Guass_0_1_seed1234_128000_2dplus_3000_r0.05_kk4_pool40IterRWD5_12951/tree.plk"
                        #------------------------------
                        # "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_2d_r0.83_kk4_pool40IterRWD5_3973/tree.plk",
                        # "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_2d_r0.78_kk4_pool40IterRWD5_8226/tree.plk",
                        # "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_2d_r0.752_kk4_pool40IterRWD5_12235/tree.plk",
                       #------------------3D----------
                    #    "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_3ddim_r1.004_kk4_pool40IterRWD5_4233/tree.plk",
                    #    "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_3ddim_r0.945_kk4_pool40IterRWD5_8219/tree.plk",
                    #    "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_3ddim_r0.9_kk4_pool40IterRWD5_12472/tree.plk"
                        
                        "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_3ddim_r0.9_kk4_pool192IterRWD5_13181/new/tree.plk",
                        "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_3ddim_r0.945_kk4_pool192IterRWD5_7817/new/tree.plk",
                        "my_dataCO/DIFCUSO_data/tsp100_3ddim_plus3000/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_3ddim_r1.004_kk4_pool192IterRWD5_4321/new/tree.plk"
                        ]

for myTreeFilename in myTreeFilename_list:
    for org_test_tspFilename in org_test_tspFilename_list:
        tree_mode = 'org'
        my_align_all_testData(myTreeFilename,org_test_tspFilename,N,top_num,maxPoolNum,point_dim,tree_mode=tree_mode)
        current_time1 = time.strftime("%H:%M:%S", time.localtime())    
        print("current_time0 = ",current_time0)
        print("current_time1 = ",current_time1)







