"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import os
import numpy as np
from numpy import array
from scipy.spatial.distance import pdist, squareform
import lkh,vrplib,requests
import subprocess
import multiprocessing,time
from multiprocessing import Pool
SCALE = 1e7


def create_problem_file(instance_num, nodes, demands, capacity, working_dir):
    os.makedirs(working_dir,exist_ok=True)
    with open(os.path.join(working_dir, str(instance_num) + ".vrp"), "w") as file:
        file.write("NAME : " + str(instance_num) + "\n")
        file.write("COMMENT : generated instance No. " + str(instance_num) + "\n")
        file.write("TYPE : CVRP\n")
        file.write("DIMENSION : " + str(len(nodes)) + "\n")
        file.write("EDGE_WEIGHT_TYPE : EUC_2D \n")
        file.write("CAPACITY : " + str(capacity) + " \n")
        file.write("NODE_COORD_SECTION\n")

        for i, node in enumerate(nodes):
            file.write(" " + str(i+1) + " " + str(int(node[0] * SCALE)) + " " + str(int(node[1] * SCALE)) + "\n")
        file.write("DEMAND_SECTION\n")
        for i, demand in enumerate(demands):
            file.write(str(i+1) + " " + str(demand) + "\n")
        file.write("DEPOT_SECTION \n 1 \n -1 \nEOF ")
        file.close()



def create_parameter_file(instance_num, working_dir, num_runs, time_limit):
    with open(os.path.join(working_dir, str(instance_num) + ".par"), "w") as file:
        file.write("PROBLEM_FILE = " + os.path.join(working_dir, str(instance_num) + ".vrp\n"))
        file.write("RUNS = " + str(num_runs) + "\n")
        if time_limit > 0:
            file.write("TIME_LIMIT = " + str(time_limit) + "\n")
        file.write("TOUR_FILE = " + os.path.join(working_dir, str(instance_num) + ".sol\n"))




# def solve_vrp_with_lkh(lkh_executable, par_file):
#     try:
#         subprocess.run([lkh_executable, par_file], check=True)
#         print(f"LKH已成功运行。")
#     except subprocess.CalledProcessError as e:
#         print(f"LKH运行出错: {e}")



def solve_vrp_with_lkh(lkh_executable,vrp_file,sol_file,coords):
    # vrp_file = "/cat_nips24_cvrp/tmp/0.vrp"
    with open(vrp_file, 'r') as file:
        problem_str = file.readlines()
    problem_str = "\n".join(problem_str)
    problem = lkh.LKHProblem.parse(problem_str)
    # solver_path = "catNips2024Code_0313/LKH-3.0.6/LKH"
    routes = lkh.solve(lkh_executable, problem=problem, max_trials=1, runs=10)
    indexs = [list(array(ll)-1) for ll in routes]
    print("==========",sum([len(l) for l in indexs]))
    print("==========",max([max(ll) for ll in indexs]))
    cost = 0
    for inds in indexs:
        tmp_num = len(inds)
        inds = [0] + inds
        for i in range(tmp_num):
            t_ind1,t_ind2 = inds[i], inds[i+1]
            cost = cost +  np.linalg.norm(coords[t_ind1] - coords[t_ind2])
   
    route_num = len(routes)
    with open(sol_file, 'w') as f:
        for row,ii in zip(routes,range(1,route_num+1)):
            tmp = ["Route #" + str(ii) + ":"]
            row = tmp + row
            f.write(" ".join(map(str, row)) + "\n")
        cost_row = ["Cost",cost]
        f.write(" ".join(map(str, cost_row)) + "\n")
    # with open(sol_file, 'w') as f:
    #     for route in routes:
    #         f.write(str(route) + '\n')
    #     f.write(str(cost) + '\n')
    # print(routes)





# vrp_file = "/cat_nips24_cvrp/tmp/5.vrp"
# with open(vrp_file, 'r') as file:
#     problem_str = file.readlines()
    
# problem_str = "\n".join(problem_str)

# problem = lkh.LKHProblem.parse(problem_str)

# solver_path = "catNips2024Code_0313/LKH-3.0.6/LKH"
# res = lkh.solve(solver_path, problem=problem, max_trials=1, runs=10)
# print("==========",max([max(ll) for ll in res]))




def myPool_genCVRP3(instance_num):
    sol_file = os.path.abspath(args.working_dir + str(instance_num) + ".sol")
    while not os.path.exists(sol_file):
        # coords = np.random.rand(args.num_nodes + 1, 2)
        coords = np.random.normal(loc=args.my_mean, scale=args.my_std, size=(args.num_nodes + 1, 2)) % 1
        demands = np.array([0] + np.random.randint(1, 10, args.num_nodes).tolist())
        create_problem_file(instance_num, coords, demands, args.capacity, args.working_dir)
        create_parameter_file(instance_num, args.working_dir, num_runs=args.num_runs,   time_limit=args.time_limit)
        lkh_cmd = os.path.join(args.lkh_exec) + ' ' + os.path.join(args.working_dir, str(instance_num) + ".par")
        os.system(lkh_cmd)
        vrp_file = os.path.abspath(args.working_dir + str(instance_num) + ".vrp")
        lkh_executable = os.path.abspath("catNips2024Code_0313/LKH-3.0.6/LKH")
        solve_vrp_with_lkh(lkh_executable,vrp_file,sol_file,coords)
        print("")
        





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate and solve CVRP")
    parser.add_argument("--num_instances", type=int, default=1280, help="Number instances")
    parser.add_argument("--num_nodes", type=int, default=100, help="Number of nodes")
    parser.add_argument("--capacity", type=int, default=50, help="Capacity")
    # parser.add_argument("--working_dir", type=str,required=True)
    parser.add_argument("--working_dir", type=str, default="./tmp/")
    parser.add_argument("--num_runs", type=int, default=10, help="LKH num runs")
    parser.add_argument("--time_limit", type=int, default=10, help="LKH time limit")
    # parser.add_argument("--lkh_exec", type=str, required=True, help="Path to LKH solver")
    parser.add_argument("--lkh_exec", type=str, default="catNips2024Code_0313/LKH-3.0.6", help="Path to LKH solver")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--output_filename", type=str,required=True)
    parser.add_argument("--output_filename", type=str,default="./haha")
    parser.add_argument("--reorder", dest="reorder", action="store_true",
                        help="Reorder nodes/tours. Must be reordered in training dataset")
    
    
    
    parser.add_argument("--myPoolNum",type=float,default=96)
    parser.add_argument("--my_mean",type=float,default=0)
    parser.add_argument("--my_std",type=float,default=100000)


    args = parser.parse_args()

    np.random.seed(args.seed)
    # all_coords, all_demands, all_capacities, all_remaining_capacities = list(), list(), list(), list()
    # all_via_depots, all_tour_lens = list(), list()
    current_time0 = time.strftime("%H:%M:%S", time.localtime())
    arg_list = range(args.num_instances)
    # arg_list = range(125000,128000)
    # arg_list = range(10)

    # with Pool(args.myPoolNum) as pool:
    #     pool.map(myPool_genCVRP3,arg_list)
    for arg in arg_list:
        myPool_genCVRP3(arg)
    current_time1 = time.strftime("%H:%M:%S", time.localtime())
    print("="*50)
    print("saved in ",args.working_dir)
    print("Guass-mean-std",args.my_mean,args.my_std)
    print("current_time0 = ",current_time0)
    print("current_time1 = ",current_time1)
    
    
    
    
    # for instance_num in range(args.num_instances):
    #     sol_file = os.path.abspath(args.working_dir + str(instance_num) + ".sol")
    #     while not os.path.exists(sol_file):
    #         print("-------------")
    #         coords = np.random.rand(args.num_nodes + 1, 2)
    #         # coords = np.random.normal(loc=args.my_mean, scale=args.my_std, size=(args.num_nodes + 1, 2)) % 10
    #         demands = np.array([0] + np.random.randint(1, 10, args.num_nodes).tolist())
    #         create_problem_file(instance_num, coords, demands, args.capacity, args.working_dir)
    #         create_parameter_file(instance_num, args.working_dir, num_runs=args.num_runs,   time_limit=args.time_limit)
    #         lkh_cmd = os.path.join(args.lkh_exec) + ' ' + os.path.join(args.working_dir, str(instance_num) + ".par")
    #         os.system(lkh_cmd)
    #         par_file = os.path.abspath(args.working_dir + str(instance_num) + ".par")
    #         lkh_executable = os.path.abspath("catNips2024Code_0313/LKH-3.0.6/LKH")
    #         solve_vrp_with_lkh(lkh_executable, par_file)
