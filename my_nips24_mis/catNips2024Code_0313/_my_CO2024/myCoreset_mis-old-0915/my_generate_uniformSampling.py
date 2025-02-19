import numpy as np
import pickle
import os,sys,glob,shutil
#tmp_cfd = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append("./")

from catNips2024Code_0313.DIFUSCO_main.difusco.co_datasets.tsp_graph_dataset import TSPGraphDataset



def my_generate_uniform_sampling(global_misFilename_list,uniform_sampling_size,seed,):
    my_samples_misFilename_list = np.random.choice(global_misFilename_list, size=uniform_sampling_size, replace=False) 
    save_path = "my_dataCO/DIFCUSO_data/mis_er/US_coreset/US_" + str(uniform_sampling_size)
    os.makedirs(save_path,exist_ok=True)
    for cf in my_samples_misFilename_list:
        shutil.copy2(cf,save_path)
        
    coreset_misFilename_result_list = [os.path.dirname(cf) + "/train_annotations/" + os.path.basename(cf)[:-8] + "_unweighted.result"  for cf in my_samples_misFilename_list]
    os.makedirs(save_path+"/train_annotations")
    for cf in coreset_misFilename_result_list:
        shutil.copy2(cf,save_path+"/train_annotations")
        
    coreset_misFilename_result_list = [os.path.dirname(cf) + "/preprocessed/kamis/" + os.path.basename(cf)[:-8] + "_unweighted.graph"  for cf in my_samples_misFilename_list]
    os.makedirs(save_path+"/preprocessed/kamis", exist_ok=True)
    for cf in coreset_misFilename_result_list:
        shutil.copy2(cf,save_path+"/preprocessed/kamis")
        
    print("")

seed = 1234; size_list = [3904,8272,12981]



global_misFilename_list =  glob.glob("my_dataCO/DIFCUSO_data/mis_er/train_data_3000/*.gpickle")
for uniform_sampling_size in size_list:
    my_generate_uniform_sampling(global_misFilename_list,uniform_sampling_size,seed)

print("")

