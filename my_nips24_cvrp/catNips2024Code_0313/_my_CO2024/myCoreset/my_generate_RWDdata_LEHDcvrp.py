import random,os,sys,pickle,time

#tmp_cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append("./")

import numpy as np
from numpy import array
from catNips2024Code_0313._my_CO2024.myCoreset.mytreePackage._myReadWriteCVRP_LEHD import my_read_cvrpData,my_save_CVRPData,my_save_CVRPlines
from catNips2024Code_0313._my_CO2024.myCoreset.mytreePackage.myTree_LEHDcvrp import my_RWD_coreset



# def my_generate_RWDdata(org_filepath,org_filename,dataset_size,ballRadius,kk,maxPoolNum,node_num,MAX_TOTAL_WEIGHT,point_dim,my_maxIterTimes_RWD=5):


def my_generate_RWDdata(org_filepath,org_filename,dataset_size,ballRadius,kk,maxPoolNum,node_num,MAX_TOTAL_WEIGHT,point_dim,my_maxIterTimes_RWD=5):
    current_time0 = time.strftime("%H:%M:%S", time.localtime())    
    org_tspFilename = os.path.abspath(org_filepath + org_filename+ '.txt')
    line_list,raw_data_nodes_list,raw_data_demand_list = my_read_cvrpData(org_tspFilename,dataset_size)
    org_weights_lists = np.copy(raw_data_demand_list)
    # org_weights_lists[:,0] = 
    org_weights_lists[:,0] = MAX_TOTAL_WEIGHT - np.sum(org_weights_lists,axis=1); org_weights_lists = org_weights_lists.astype(np.float32)
    org_location_lists = np.copy(raw_data_nodes_list)
    # org_location_lists = [locs[1:]-wei.dot(locs[1:]) for locs,wei in zip(raw_data_nodes_list,org_weights_lists) ]
    org_location_lists = [ locs - org_location_lists[0] for locs in org_location_lists]
    coreset_new_locations_lists,coreset_id_list,my_tree = my_RWD_coreset(org_location_lists,org_weights_lists,ballRadius,kk,maxPoolNum,maxIterTimes_RWD=my_maxIterTimes_RWD)
    mySavePath = org_filepath + 'myRWD_data/res_' + org_filename +  '_r' + str(ballRadius) + '_kk' + str(kk)+ '_pool' + str(maxPoolNum) +'IterRWD' + str(my_maxIterTimes_RWD) +'_' + str(len(coreset_id_list)) 
    if not(os.path.exists(mySavePath)):
        os.mkdir(mySavePath)
        os.mkdir(mySavePath+"/org")
        os.mkdir(mySavePath+"/new")
    #tspDataFilename_RWD = mySavePath + '/' +  org_filename + "_myRWD"
    #my_save_tspData(tspDataFilename_RWD,new_global_locations_lists,org_tour_lists)

    tspDataCoresetFilename_RWD_org = mySavePath + '/org/' +  org_filename + "_myRWDCoreset"
    tspDataCoresetFilename_RWD_new = mySavePath + '/new/' +  org_filename + "_myRWDCoreset"
    org_coreset_lines = array(line_list)[coreset_id_list]
    my_save_CVRPlines(tspDataCoresetFilename_RWD_org,org_coreset_lines)
    depot_list = np.zeros((org_coreset_lines.shape[0],point_dim))
    my_save_CVRPData(tspDataCoresetFilename_RWD_new,org_coreset_lines,depot_list,coreset_new_locations_lists,point_dim)      

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










# 17

def test_100(kk=2):
    for ballRadius in [17.5]:
        print("-----------------------")
        dataset_size = 128000; m = 100; node_num = 100; MAX_TOTAL_WEIGHT = node_num * 10

        # point_dim = 3; dim_mode="ddim"; Guass_std = 1; plus_size = 3000
        point_dim = 2; dim_mode="d"; Guass_std = 1; plus_size = 3000

        org_filepath = "my_dataCO/DIFCUSO_data/tsp100_" + str(point_dim) + dim_mode + "_plus" + str(plus_size) + '/'
        org_filename = 'tsp' + str(m) + "_train_Guass_0_" + str(Guass_std) + "_seed1234_" + str(dataset_size) + "_" + str(point_dim) + dim_mode + "plus_" + str(plus_size)
        org_filename = 'tsp' + str(m) + "_train_Uniform_seed1234_" + str(dataset_size) + "_" + str(point_dim) + dim_mode
        org_filepath = "my_dataCO/LEHD_data/"
        org_filename = "cvrp_0_100000_128t"
        maxPoolNum = 64
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