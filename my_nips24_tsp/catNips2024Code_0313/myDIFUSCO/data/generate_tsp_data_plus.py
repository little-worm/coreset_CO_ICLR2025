import random,os,sys,pickle,time
from multiprocessing import Pool
sys.path.append("./")
from my_global import *
import numpy as np
from numpy import ones
from catNips2024Code_0313._my_CO2024.myCoreset.mytreePackage.myReadWriteTSP import my_read_tspData,my_save_tspData

org_filepath = "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000/"
# org_filepath = "my_dataCO/DIFCUSO_data/tsp100_3ddim/"
org_filename_list = [ # "tsp100_train_Uniform_seed1234_128000_2d.txt",
                      "tsp100_train_Guass_0_1_seed1234_128000_2d.txt",
                      # "tsp100_train_Guass_0_2_seed1234_128000_2d.txt",
                      # "tsp100_train_Guass_0_3_seed1234_128000_2d.txt",
                      # "tsp100_train_Guass_0_4_seed1234_128000_2d.txt",
                      # "tsp100_train_Guass_0_8_seed1234_128000_2d.txt"
                      #------------------------------------------
                      # "tsp100_train_Guass_0_1_seed1234_128000_3ddim.txt"
                    #   "tsp100_test_Uniform_seed666_1280_3ddim.txt"
                    ]

plus_filename_list = ["tsp100_train_Uniform_seed1234_128000_2d.txt"
                    #  "tsp100_train_Uniform_seed1234_128000_3ddim.txt"
                      ]
point_dim = 2
# point_dim = 3
plus_size = 3000
data_size = 128000

for org_filename in org_filename_list:
    data_list,tour_list = my_read_tspData(org_filepath+org_filename,data_size)
    for i,fn in zip(range(len(plus_filename_list)),plus_filename_list):
        id = round(i*plus_size)
        tmp_data_list,tmp_tours_list = my_read_tspData(org_filepath+fn,data_size)
        data_list[id:id+plus_size] = tmp_data_list[id:id+plus_size]
        tour_list[id:id+plus_size] = tmp_tours_list[id:id+plus_size]
        print("")
    save_filename = org_filename[0:-4] + "plus_" + str(plus_size) + ".txt"
    my_save_tspData(org_filepath+save_filename,data_list,tour_list,point_dim)


    