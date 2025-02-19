import os
import sys
import numpy as np
from tqdm import tqdm
from numpy import array
sys.path.append(os.path.abspath("../../.."))
print(sys.path.append)





def tow_col_nodeflag(node_flag):
    tow_col_node_flag = []
    V = int(len(node_flag) / 2)
    for i in range(V):
        tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
    return tow_col_node_flag







def my_read_cvrpData(org_tspFilename,n):
    raw_data_nodes_list = []
    raw_data_capacity_list = []
    raw_data_demand_list = []
    raw_data_cost_list = []
    raw_data_node_flag_list = []
    line_list = []; depot_list = []; customer_list = []
    for line in tqdm(open(org_tspFilename, "r").readlines()[:n], ascii=True):
        line = line.split(",")
        line_list.append(line)
        depot_index = int(line.index('depot'))
        customer_index = int(line.index('customer'))
        capacity_index = int(line.index('capacity'))
        demand_index = int(line.index('demand'))
        cost_index = int(line.index('cost'))
        node_flag_index = int(line.index('node_flag'))

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        depot_list.append(depot)
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]
        customer_list.append(customer)
        loc = depot + customer
        capacity = int(float(line[capacity_index + 1]))
        if int(line[demand_index + 1]) ==0:
            demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
        else:
            demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

        cost = float(line[cost_index + 1])
        node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]

        node_flag = tow_col_nodeflag(node_flag)

        raw_data_nodes_list.append(loc)
        raw_data_capacity_list.append(capacity)
        raw_data_demand_list.append(demand)
        raw_data_cost_list.append(cost)
        raw_data_node_flag_list.append(node_flag)
    return line_list,raw_data_nodes_list,raw_data_demand_list








def my_save_CVRPData(save_filename,line_list,depot_list,all_locations_list,dim=2):
    new_line_list = []
    for line,depot,all_locations in zip(line_list,depot_list,all_locations_list):
        line = array(line)
        cvrp_size = int((line.shape[0]-8-dim) / 5)
        line[1:1+dim] = array(depot)
        line[2+dim:2+dim+cvrp_size*dim] = array(all_locations[1:]).flatten()
        new_line_list.append(line)

    with open(save_filename, 'w') as f:
        for line in new_line_list:
            f.write(','.join(line))
    # print(line[:100])   

     
def my_save_CVRPlines(save_filename,lines_list):
    with open(save_filename, 'w') as f:
        for line in lines_list:
            f.write(','.join(line))
    # print(line[:100])   
     
     
     
     
        
# print("")
# org_tspFilename = "/cat_nips24_cvrp/my_dataCO/LEHD_data/vrp200_test_lkh.txt"; n=128
# res = my_read_cvrpData(org_tspFilename,n)

# save_filename = './hahah_output.txt'
# line_list,raw_data_nodes_list,raw_data_demand_list = res
# depot_list = [dl[0] for dl in raw_data_nodes_list]
# all_locations_list = raw_data_nodes_list
# dim = 2
        
        
# my_save_CVRPData(save_filename,line_list,depot_list,all_locations_list,dim=dim)
        


# print("")





















