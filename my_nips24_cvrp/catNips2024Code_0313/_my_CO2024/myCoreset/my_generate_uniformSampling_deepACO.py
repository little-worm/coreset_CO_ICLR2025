import numpy as np
import pickle
import os,sys
#tmp_cfd = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append("./")




def my_generate_uniform_sampling(filepath,Filename,uniform_sampling_size,seed):
    org_file = os.path.abspath(filepath + '/' + Filename)
    uniform_sampling_Filename = os.path.abspath(filepath + '/' + 'myUS_data/' + Filename[0:-4] + '_' + str(uniform_sampling_size) + '_seed' + str(seed)) + '.txt'
    with open(org_file, 'r') as file:
        # Read each line, convert it to a list of floats
        data_list = [list(map(float, line.strip().split(','))) for line in file.readlines()]
    my_samples_indexs = list(np.random.choice(np.arange(len(data_list)), size=uniform_sampling_size, replace=False))
    my_samples = np.array(data_list)[my_samples_indexs] 
    print("uniform_sampling_Filename = ",uniform_sampling_Filename)
    with open(uniform_sampling_Filename, 'w') as f:
        for da in my_samples:
            f.write(",".join(map(str, da)) + "\n")

    



# seed = 1234; point_dim = '2'; my_Guass_std="1"; size_list = [16938]; dim_mode="d"; plus_size = 3000
#----------3D---------
# seed = 1234; point_dim = '2'; size_list = [3973,8226,12235]; dim_mode="d"; plus_size = 3000

# Uniform
# seed = 1234; point_dim = '3'; size_list = [4233,8219,12472]; dim_mode="ddim"; plus_size = 3000

seed = 1234; size_list = [4697,7694,13908]


# filepath = "my_dataCO/DIFCUSO_data/tsp100_" + point_dim + dim_mode + "_plus" + str(plus_size)
# Filename = "tsp100_train_Guass_0_" + my_Guass_std + "_seed1234_128000_" + point_dim + dim_mode + "plus_" + str(plus_size) + ".txt"
# Filename = "tsp100_train_Uniform_seed1234_128000_" + point_dim + dim_mode + ".txt"
###   print("filepath = ",filepath)
###   print("Filename = ",Filename)
filepath = "/cat_nips24_cvrp/my_dataCO/deepNCO_data/cvrp/train_data/"
Filename = "cvrp100_0_10000_125000_3000.txt"
for uniform_sampling_size in size_list:
    my_generate_uniform_sampling(filepath,Filename,uniform_sampling_size,seed)



