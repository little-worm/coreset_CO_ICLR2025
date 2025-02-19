import tsplib95
import os
import gzip
import tempfile


import argparse,sys
import pprint as pp
import time
import warnings
from multiprocessing import Pool
sys.path.append('catNips2024Code_0313/DIFUSCO_main/data/pyconcorde')
sys.path.append('./')

import lkh
import numpy as np
import tqdm
#from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde
from catNips2024Code_0313._my_CO2024.myCoreset.mytreePackage.myReadWriteTSP import my_read_tspData,my_save_tspData

warnings.filterwarnings("ignore")


def solve_tsp(nodes_coord,solver="lkh",lkh_trails=1000):
  num_nodes = np.array(nodes_coord).shape[0]
  if solver == "concorde":
    pass
    #scale = 1e6
    #solver = TSPSolver.from_data(nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="EUC_2D")
    #solution = solver.solve(verbose=False)
    #tour = solution.tour
  elif solver == "lkh":
    scale = 1e6
    ## my
    ## lkh_path = 'LKH-3.0.6/LKH'
    #lkh_path = '/remote-home/share/worm/wormICML2024Code/LKH-3.0.6/LKH'
    lkh_path = "catNips2024Code_0313/LKH-3.0.6/LKH"
    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'
    problem.type = 'TSP'
    problem.dimension = num_nodes
    problem.edge_weight_type = 'EUC_2D'
    problem.node_coords = {n + 1: nodes_coord[n] * scale for n in range(num_nodes)}
    solution = lkh.solve(lkh_path, problem=problem, max_trials=lkh_trails, runs=10)
    tour = [n - 1 for n in solution[0]]
  else:
    raise ValueError(f"Unknown solver: {solver}")
  return tour









def read_TSPlib_dataset(directory,dim_threshold_min=0,dim_threshold_max=10000,dim=2):
    tsp_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tsp.gz')]
    euclidean_coords = []

    for tsp_file in tsp_files:
        with gzip.open(tsp_file, 'rb') as f_in:
            with tempfile.NamedTemporaryFile(delete=False) as f_out:
                f_out.write(f_in.read())
                temp_tsp_file = f_out.name
        problem = tsplib95.load(temp_tsp_file)
        print(problem.edge_weight_type)
        os.remove(temp_tsp_file)  # 删除临时文件
        if problem.edge_weight_type == 'EUC_' + str(dim) + 'D' and problem.dimension > dim_threshold_min and problem.dimension < dim_threshold_max:
            problem_dict_coords = problem.node_coords
            problem_coords = [problem_dict_coords[n] for n in problem_dict_coords.keys()]
            euclidean_coords.append(problem_coords)
            print(f"File {tsp_file} is Euclidean-2D, dimension = {problem.dimension} ")
        else:
            pass
            # print(f"File {tsp_file} is not Euclidean-2D")

    return euclidean_coords


dim = 3
directory = "my_dataCO/ALL_tsp"  # Replace with the path to your directory containing .tsp.gz files
dim_threshold_min=100; dim_threshold_max=5000
euclidean_coords = read_TSPlib_dataset(directory,dim_threshold_min,dim_threshold_max,dim=dim)
print(f"Found {len(euclidean_coords)} Euclidean TSP problems")

tour_list = []; myTspData_list = []
for data in euclidean_coords:
    data = np.array(data)
    max_data = data.max(axis=0)
    min_data = data.min(axis=0)
    scale_data = max_data - min_data
    data = data / scale_data *10
    data = data - data.mean(axis=0)
    # print("data = ",data)
    tour = solve_tsp(data)
    tour_list.append(tour)
    myTspData_list.append(data)
    # print(tour)
save_filepathname = "my_dataCO/DIFCUSO_data/tsplib/my_tsblib_Eud" + str(dim) + "d_threshod"+str(dim_threshold_min)+"-"+ str(dim_threshold_max) +".txt"
print("===================================")
my_save_tspData(save_filepathname,myTspData_list,tour_list,point_dim=2)
print("===================================")












