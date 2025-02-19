import random,os,sys,pickle,time

#tmp_cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append("./")
sys.path.append("catNips2024Code_0313/myDeepACO/cvrp")

import numpy as np
from numpy import array
# from catNips2024Code_0313._my_CO2024.myCoreset.mytreePackage._myReadWriteCVRP_LEHD import my_read_cvrpData,my_save_CVRPData,my_save_CVRPlines
from mytreePackage.myTree_deepACO import my_RWD_coreset_deepACO_cvrp
from mytreePackage.myReadWriteCVRP_deepNCO import my_readData_deepNCO,my_saveData_deepNCO
# from my_gen_cvrp_org_dataset import my_readData_deepNCO


# def my_readData_deepNCO(data_file_name,node_num,point_dim=2):
#     with open(data_file_name, 'r') as file:
#         # Read each line, convert it to a list of floats
#         data_list = [list(map(float, line.strip().split(','))) for line in file.readlines()]
#     all_locations_list = []; all_demands_list = []
#     for da in data_list:
#         tmp_locs = list(np.array(da[:point_dim*(node_num+1)]).reshape(-1,point_dim))
#         all_locations_list.append(tmp_locs)
#         all_demands_list.append(da[point_dim*(node_num+1):])
#     return array(all_locations_list),array(all_demands_list)





# def my_saveData_deepNCO(data_file_name,all_locations_list,all_demands_list,point_dim=2):
#     all_data = []
#     for all_locations,all_demands in zip(all_locations_list,all_demands_list):
#         # all_locations = np.random.normal(loc=mean, scale=100000, size=(num_nodes + 1, point_dim)) % 1
#         # all_demands = np.random.randint(1,10,size=(num_nodes+1)); all_demands[0] = 0
#         one_data = list(all_locations.ravel()) + list(all_demands)
#         all_data.append(one_data)
#     with open(data_file_name, 'w') as f:
#         for da in all_data:
#             f.write(",".join(map(str, da)) + "\n")







def my_generate_RWDdata(org_filepath,org_filename,dataset_size,ballRadius,kk,maxPoolNum,node_num,MAX_TOTAL_WEIGHT,point_dim,my_maxIterTimes_RWD=5):
    current_time0 = time.strftime("%H:%M:%S", time.localtime())    
    data_file_name = os.path.abspath(org_filepath + org_filename+ '.txt')
    all_locations_list,all_demands_list = my_readData_deepNCO(data_file_name,node_num,point_dim)
    # line_list,raw_data_nodes_list,raw_data_demand_list = my_read_cvrpData(org_tspFilename,dataset_size)
    # org_weights_lists = [array(wei[1:])/sum(wei[1:]) for wei in raw_data_demand_list]
    # org_location_lists = [locs[1:]-wei.dot(locs[1:]) for locs,wei in zip(raw_data_nodes_list,org_weights_lists) ]
    org_weights_lists = np.copy(all_demands_list)
    # my_del1 = np.sum(org_weights_lists,axis=1)
    org_weights_lists[:,0] = MAX_TOTAL_WEIGHT - np.sum(org_weights_lists,axis=1)
    # my_del2 = np.sum(org_weights_lists,axis=1)
    org_location_lists = [ locs - all_locations_list[0] for locs in all_locations_list]
    coreset_new_locations_lists,coreset_id_list,my_tree = my_RWD_coreset_deepACO_cvrp(org_location_lists,org_weights_lists,ballRadius,kk,maxPoolNum,maxIterTimes_RWD=my_maxIterTimes_RWD)
    mySavePath = org_filepath + 'myRWD_data/res_' + org_filename +  '_r' + str(ballRadius) + '_kk' + str(kk)+ '_pool' + str(maxPoolNum) +'IterRWD' + str(my_maxIterTimes_RWD) +'_' + str(len(coreset_id_list)) 
    if not(os.path.exists(mySavePath)):
        os.mkdir(mySavePath)
        os.mkdir(mySavePath+"/org")
        os.mkdir(mySavePath+"/new")
    # tspDataFilename_RWD = mySavePath + '/' +  org_filename + "_myRWD"
    #my_save_tspData(tspDataFilename_RWD,new_global_locations_lists,org_tour_lists)
    # my_saveData_deepNCO(tspDataFilename_RWD,)
    tspDataCoresetFilename_RWD_org = mySavePath + '/org/' +  org_filename + "_myRWDCoreset"
    tspDataCoresetFilename_RWD_new = mySavePath + '/new/' +  org_filename + "_myRWDCoreset"
    
    my_saveData_deepNCO(tspDataCoresetFilename_RWD_org,all_locations_list[coreset_id_list],all_demands_list[coreset_id_list])
    my_saveData_deepNCO(tspDataCoresetFilename_RWD_new,coreset_new_locations_lists,all_demands_list[coreset_id_list])
    # org_coreset_lines = array(line_list)[coreset_id_list]
    # my_save_CVRPlines(tspDataCoresetFilename_RWD_org,org_coreset_lines)
    
    # my_save_CVRPData(tspDataCoresetFilename_RWD_new,org_coreset_lines,np.zeros(point_dim),coreset_new_locations_lists,point_dim)      

    #ids_weights_Filename = mySavePath + '/' + 'id_weights'

    #np.save(ids_weights_Filename, [[child_id_flatenList],[child_weight_flatenList]])
    mySaveTreeFilename_org = mySavePath + '/org/' + 'tree.plk'
    mySaveTreeFilename_new = mySavePath + '/new/' + 'tree.plk'
    with open(mySaveTreeFilename_org, 'wb') as file:
        pickle.dump(my_tree, file)
    with open(mySaveTreeFilename_new, 'wb') as file:
        pickle.dump(my_tree, file)
    current_time1 = time.strftime("%H:%M:%S", time.localtime())    
    print('mySavePath = ',mySavePath)
    print("current_time0 = ",current_time0)
    print("current_time1 = ",current_time1)
    print("-------------------end ------------------end --------------------------------------")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")












def test_100(kk=2):
    for ballRadius in [18]:
        print("-----------------------")
        dataset_size = 128000; m = 100; node_num = 100; MAX_TOTAL_WEIGHT = node_num * 10
        point_dim = 2; dim_mode="d"; Guass_std = 1; plus_size = 3000
        # org_filepath = "my_dataCO/DIFCUSO_data/tsp100_" + str(point_dim) + dim_mode + "_plus" + str(plus_size) + '/'
        # org_filename = 'tsp' + str(m) + "_train_Guass_0_" + str(Guass_std) + "_seed1234_" + str(dataset_size) + "_" + str(point_dim) + dim_mode + "plus_" + str(plus_size)
        # org_filename = 'tsp' + str(m) + "_train_Uniform_seed1234_" + str(dataset_size) + "_" + str(point_dim) + dim_mode
        # org_filepath = "my_dataCO/LEHD_data/"
        org_filename = "cvrp100_0_10000_125000_3000"
        # org_filename = "cvrp100_0_10000_1250_30"
        org_filepath = "my_dataCO/deepNCO_data/cvrp/train_data/"
        # org_filename = "vrp200_test_lkh"
        maxPoolNum = 40
        print(org_filepath)
        print(org_filename)
        my_generate_RWDdata(org_filepath,org_filename,dataset_size,ballRadius,kk,maxPoolNum,node_num,MAX_TOTAL_WEIGHT,point_dim)


#test_100(kk=2) 
test_100(4)                     
#test_100(kk=8)
#test_100(kk=16)





#   loaded_matrix = np.load('matrix_file')
#   print(loaded_matrix)



#   with open(Filename, 'rb') as file:
#       loaded_tree = pickle.load(file)
#   print("loaded_tree = ",loaded_tree)















