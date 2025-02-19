"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import lkh,vrplib,requests
import subprocess

SCALE = 1e7

all_one_row_data = []




def my_lineCVRP(cvrp_file_path,instance_size):
    save_file_name = cvrp_file_path[:-1] + '.txt'
    for instance_num in range(instance_size):
        if instance_num % 1000 == 0:
            print("instance_num = ",instance_num)
        vrp_file_name = cvrp_file_path + str(instance_num) + ".vrp"
        sol_file_name = cvrp_file_path + str(instance_num) + ".sol"
        instance = vrplib.read_instance(vrp_file_name)
        solution = vrplib.read_solution(sol_file_name)

        scalar = np.max(instance["node_coord"])
        depot = instance["node_coord"][0] / scalar
        Customer = instance["node_coord"][1:] / scalar
        demand = instance["demand"]
        capacity = instance["capacity"]

        depot = np.array(depot).ravel().tolist()
        Customer = np.array(Customer).ravel().tolist()
        demand = np.array(demand).ravel().tolist()

        cost = solution["cost"]

        routes = solution["routes"]
        tmp_flag_list = []; reval_routes = []
        for ll in routes:
            tmp = [0 for i in range(len(ll))]
            tmp[0] = 1
            tmp_flag_list = tmp_flag_list + tmp
            reval_routes = reval_routes + ll
        reval_routes = list(np.array(reval_routes) - 1)
        curr_node_flag = reval_routes + tmp_flag_list
        one_row_data = ['depot'] + depot + ['customer'] + Customer + ['capacity'] + [capacity] +  ['demand'] + demand + ['cost'] + [cost] + ['node_flag'] + curr_node_flag 
        all_one_row_data.append(one_row_data)


    with open(save_file_name, 'w') as f:
        for row in all_one_row_data:
            f.write(",".join(map(str, row)) + "\n")






cvrp_file_path = "tmp/"; instance_size = 10
cvrp_file_path = "cvrp_0_100000_128t/"; instance_size = 128000
# cvrp_file_path = "cvrp_0_025_125t/"; instance_size = 128000
# cvrp_file_path = "cvrp_0_1_125t/"; instance_size = 128000
# cvrp_file_path = "cvrp_0_100000_1280/"; instance_size = 1280

my_lineCVRP(cvrp_file_path,instance_size)

















