import os
import sys
import numpy as np
from tqdm import tqdm
from numpy import array
sys.path.append(os.path.abspath("../../.."))
print(sys.path.append)




def my_readData_deepNCO(data_file_name,node_num,point_dim=2):
    with open(data_file_name, 'r') as file:
        # Read each line, convert it to a list of floats
        data_list = [list(map(float, line.strip().split(','))) for line in file.readlines()]
    all_locations_list = []; all_demands_list = []
    for da in data_list:
        tmp_locs = list(np.array(da[:point_dim*(node_num+1)]).reshape(-1,point_dim))
        all_locations_list.append(tmp_locs)
        all_demands_list.append(da[point_dim*(node_num+1):])
    return array(all_locations_list),array(all_demands_list)





def my_saveData_deepNCO(data_file_name,all_locations_list,all_demands_list,point_dim=2):
    all_data = []
    for all_locations,all_demands in zip(all_locations_list,all_demands_list):
        # all_locations = np.random.normal(loc=mean, scale=100000, size=(num_nodes + 1, point_dim)) % 1
        # all_demands = np.random.randint(1,10,size=(num_nodes+1)); all_demands[0] = 0
        one_data = list(all_locations.ravel()) + list(all_demands)
        all_data.append(one_data)
    with open(data_file_name, 'w') as f:
        for da in all_data:
            f.write(",".join(map(str, da)) + "\n")


































