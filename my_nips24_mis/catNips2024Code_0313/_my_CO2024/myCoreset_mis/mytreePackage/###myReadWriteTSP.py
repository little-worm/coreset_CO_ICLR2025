import os
import sys
import numpy as np

sys.path.append("./")

from  my_global import *
from DIFUSCO_main.difusco.co_datasets.tsp_graph_dataset import TSPGraphDataset




def my_read_tspData(Filename,n):
    myTspdata = TSPGraphDataset(Filename)
    myTspData_list = []
    myTspTour_list = []
    for idx in range(n):
        tspData,tour =  myTspdata.get_example(idx)
        myTspData_list.append(tspData)
        myTspTour_list.append(tour)
    print(len(myTspData_list),len(myTspData_list[0]),len(myTspTour_list))
    return myTspData_list,myTspTour_list





#   Filename = "my_dataCO/DIFCUSO_data/tsp/tsp100_test_Guass_0_0.5_seed666_1280.txt"
#   n = 128  
#   myTspData_list,myTspTour_list = my_read_tspData(Filename,n)






def my_save_tspData(Filename,myTspData_list,myTspTour_list,point_dim):
    f = open(Filename, "w")
    for data, tour in zip(myTspData_list,myTspTour_list):
        if point_dim==2:
            f.write(" ".join(str(x) + str(" ") + str(y) for x, y in data))
        elif point_dim==3:
            f.write(" ".join(str(x) + str(" ") + str(y) + str(" ") + str(z) for x, y, z in data))
        else:
            assert 0,"undefined point_dim!!!"
        f.write(str(" ") + str('output') + str(" "))
        f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
        f.write(str(" ") + str(tour[0] + 1) + str(" "))
        f.write("\n")




def test():
    Filename = 'haha'
    n=3; m=3; d=2
    myTspData_list = np.random.rand(n,m,d)
    myTspTour_list = np.random.rand(n,m)
    my_save_tspData(Filename,myTspData_list,myTspTour_list)



#   test()




