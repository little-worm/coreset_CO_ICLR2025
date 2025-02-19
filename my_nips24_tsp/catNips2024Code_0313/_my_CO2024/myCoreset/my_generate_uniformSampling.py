import numpy as np
import pickle
import os,sys
#tmp_cfd = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append("./")

from catNips2024Code_0313.myDIFUSCO.difusco.co_datasets.tsp_graph_dataset import TSPGraphDataset



def my_generate_uniform_sampling(filepath,Filename,uniform_sampling_size,seed):
    org_file = os.path.abspath(filepath + '/' + Filename)
    uniform_sampling_Filename = os.path.abspath(filepath + '/' + 'myUniformSample_data/' + Filename[0:-4] + '_' + str(uniform_sampling_size) + '_seed' + str(seed)) + '.txt'
    myTspdata = TSPGraphDataset(org_file)
    data = myTspdata.file_lines
    my_samples = np.random.choice(data, size=uniform_sampling_size, replace=False) 
    print("uniform_sampling_Filename = ",uniform_sampling_Filename)
    np.savetxt(uniform_sampling_Filename,my_samples,fmt='%s')



# seed = 1234; point_dim = '2'; my_Guass_std="1"; size_list = [16938]; dim_mode="d"; plus_size = 3000
#----------3D---------
# seed = 1234; point_dim = '2'; size_list = [3973,8226,12235]; dim_mode="d"; plus_size = 3000

# Uniform
seed = 1234; point_dim = '3'; size_list = [4321,7817,13181]; dim_mode="ddim"; plus_size = 3000



filepath = "my_dataCO/DIFCUSO_data/tsp100_" + point_dim + dim_mode + "_plus" + str(plus_size)
# Filename = "tsp100_train_Guass_0_" + my_Guass_std + "_seed1234_128000_" + point_dim + dim_mode + "plus_" + str(plus_size) + ".txt"
Filename = "tsp100_train_Uniform_seed1234_128000_" + point_dim + dim_mode + ".txt"
###   print("filepath = ",filepath)
###   print("Filename = ",Filename)


for uniform_sampling_size in size_list:
    my_generate_uniform_sampling(filepath,Filename,uniform_sampling_size,seed)



