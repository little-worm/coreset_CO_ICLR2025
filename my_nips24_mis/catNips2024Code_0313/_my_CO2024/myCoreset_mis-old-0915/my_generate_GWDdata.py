import random,os,sys,pickle,time,glob,shutil

#tmp_cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append("./")

import numpy as np
from mytreePackage.myTree_mis import my_RWD_coreset




def my_generate_RWDdata(org_filepath,org_filename,kk,maxPoolNum,ballRadius_RWD=0.052,ballRadius_pointset=1,point_num_threshold=1000):
    current_time0 = time.strftime("%H:%M:%S", time.localtime())    

    global_misFilename_list =  glob.glob("my_dataCO/DIFCUSO_data/mis_er/train_data_3000/*.gpickle")
    print("global_misFilename_list = ",len(global_misFilename_list))
    coreset_misFilename_list = my_RWD_coreset(global_misFilename_list,ballRadius_RWD,kk,maxPoolNum,ballRadius_pointset,point_num_threshold)

    mySavePath = org_filepath + '/res_' + org_filename + str(len(global_misFilename_list))+  '_r' + str(ballRadius_RWD) + '_kk' + str(kk)+ '_pool' + str(maxPoolNum) + '_' + str(len(list(coreset_misFilename_list))) 
    if not(os.path.exists(mySavePath)):
        os.mkdir(mySavePath)
    #tspDataFilename_RWD = mySavePath + '/' +  org_filename + "_myRWD"
    #my_save_tspData(tspDataFilename_RWD,new_global_locations_lists,org_tour_lists)

    # my_misFilename_lists = [ global_misFilename_list[t_id] for t_id in coreset_misFilename_list ]
    tspDataCoresetFilename_RWD = mySavePath + '/' +  org_filename + "_myRWDCoreset"


        
    for cf in coreset_misFilename_list:
        shutil.copy2(cf,mySavePath)
    
    if not(os.path.exists(mySavePath+"/train_annotations")):
        coreset_misFilename_result_list = [os.path.dirname(cf) + "/train_annotations/" + os.path.basename(cf)[:-8] + "_unweighted.result"  for cf in coreset_misFilename_list]
        os.mkdir(mySavePath+"/train_annotations")
    for cf in coreset_misFilename_result_list:
        shutil.copy2(cf,mySavePath+"/train_annotations")
        
        
    # if not(os.path.exists(mySavePath+"/preprocessed/kamis")):
    coreset_misFilename_result_list = [os.path.dirname(cf) + "/preprocessed/kamis/" + os.path.basename(cf)[:-8] + "_unweighted.graph"  for cf in coreset_misFilename_list]
    # os.mkdir(mySavePath+"/preprocessed/kamis")
    os.makedirs(mySavePath+"/preprocessed/kamis", exist_ok=True)
    for cf in coreset_misFilename_result_list:
        shutil.copy2(cf,mySavePath+"/preprocessed/kamis")
        
    print("")
    
        
        
        
        
        
    current_time1 = time.strftime("%H:%M:%S", time.localtime())    
    print('mySavePath = ',mySavePath)
    print("current_time0 = ",current_time0)
    print("current_time1 = ",current_time1)
    print("-------------------end ------------------end --------------------------------------")
    print("")
    print("")
    print("")










def test_100(kk=2):
    for ballRadius_RWD in [0.299,0.319]:
        print("-----------------------",ballRadius_RWD)
        org_filepath = "my_dataCO/DIFCUSO_data/mis_er/RWD_coreset" 
        org_filename = 'mis_er'
        kk=4;maxPoolNum=96;ballRadius_pointset=1;point_num_threshold=21
        my_generate_RWDdata(org_filepath,org_filename,kk,maxPoolNum,ballRadius_RWD,ballRadius_pointset,point_num_threshold)







#test_100(kk=2) 
test_100(kk=4)                     
#test_100(kk=8)
#test_100(kk=16)





# 0.38 914
# 0.35 5820


