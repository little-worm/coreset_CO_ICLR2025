import random,os,sys,pickle,time

#tmp_cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append("./")

import numpy as np
from mytreePackage.myReadWriteTSP import my_read_tspData,my_save_tspData
from mytreePackage.myTree import my_RWD_coreset



def my_generate_RWDdata(org_filepath,org_filename,n,m,ballRadius,kk,maxPoolNum,point_dim,my_maxIterTimes_RWD=5):
    current_time0 = time.strftime("%H:%M:%S", time.localtime())    
    org_tspFilename = os.path.abspath(org_filepath + org_filename+ '.txt')
    org_location_lists,org_tour_lists = my_read_tspData(org_tspFilename,n)
    org_location_lists = [ll-np.mean(ll,axis=0) for ll in org_location_lists ]
    coreset_locations_lists,coreset_tours_lists,my_tree = my_RWD_coreset(org_location_lists,org_tour_lists,ballRadius,kk,maxPoolNum,maxIterTimes_RWD=my_maxIterTimes_RWD)
    mySavePath = org_filepath + '/myRWD_data/res_' + org_filename +  '_r' + str(ballRadius) + '_kk' + str(kk)+ '_pool' + str(maxPoolNum) +'IterRWD' + str(my_maxIterTimes_RWD) +'_' + str(len(coreset_tours_lists)) 
    if not(os.path.exists(mySavePath)):
        os.mkdir(mySavePath)
    #tspDataFilename_RWD = mySavePath + '/' +  org_filename + "_myRWD"
    #my_save_tspData(tspDataFilename_RWD,new_global_locations_lists,org_tour_lists)

    tspDataCoresetFilename_RWD = mySavePath + '/' +  org_filename + "_myRWDCoreset"
    my_save_tspData(tspDataCoresetFilename_RWD,coreset_locations_lists,coreset_tours_lists,point_dim)      

    #ids_weights_Filename = mySavePath + '/' + 'id_weights'

    #np.save(ids_weights_Filename, [[child_id_flatenList],[child_weight_flatenList]])
  
    mySaveTreeFilename = mySavePath + '/' + 'tree.plk'
    with open(mySaveTreeFilename, 'wb') as file:
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









def test_01(kk=2):
    org_filename = 'test_TSP200_n128'
    n = 128; m = 200; 
    ballRadius = 0.005; maxPoolNum = 40

    my_generate_RWDdata(org_filename,n,m,ballRadius,kk,maxPoolNum)




def test_100(kk=2):
    for ballRadius in [0.752]:
        print("-----------------------")
        n = 128000; m = 100; 
        # point_dim = 3; dim_mode="ddim"; Guass_std = 1; plus_size = 3000
        point_dim = 2; dim_mode="d"; Guass_std = 1; plus_size = 3000
        org_filepath = "my_dataCO/DIFCUSO_data/tsp100_" + str(point_dim) + dim_mode + "_plus" + str(plus_size) + '/'
        org_filename = 'tsp' + str(m) + "_train_Guass_0_" + str(Guass_std) + "_seed1234_" + str(n) + "_" + str(point_dim) + dim_mode + "plus_" + str(plus_size)
        org_filename = 'tsp' + str(m) + "_train_Uniform_seed1234_" + str(n) + "_" + str(point_dim) + dim_mode
        maxPoolNum = 40
        print(org_filepath)
        print(org_filename)
        my_generate_RWDdata(org_filepath,org_filename,n,m,ballRadius,kk,maxPoolNum,point_dim)



#test_100(kk=2) 
test_100(kk=4)                     
#test_100(kk=8)
#test_100(kk=16)





#   loaded_matrix = np.load('matrix_file')
#   print(loaded_matrix)



#   with open(Filename, 'rb') as file:
#       loaded_tree = pickle.load(file)
#   print("loaded_tree = ",loaded_tree)