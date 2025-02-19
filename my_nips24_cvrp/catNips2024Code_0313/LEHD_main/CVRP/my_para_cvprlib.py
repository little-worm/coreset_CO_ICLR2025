
# # ## test------------------------------------

# data_source = "cvrp_0_100000_128t"
# data_source = "cvrp_0_025_125t"
data_source = "cvrp_0_010_125t"



# CUDA_DEVICE_NUM = 1
# my_epochs = 40; problem_size = 0

# # ================================
# method = "US"
# method = "RWD"
# # ================================
# # sample_size = 13908 # sample_size = 11428 # sample_size = 8538 # sample_size = 5885 # sample_size = 12318 # sample_size = 12551 # sample_size = 3614 # sample_size = 4314 # sample_size = 3444 # sample_size = 11755
# # sample_size = 4697
# # sample_size = 7694
# # sample_size = 12033
# #-------------------------------------------
# sample_size = 4437
# sample_size = 8082
# sample_size = 12175
# # ================================
# RRC_budget = 500
# # ================================
# path = "/cat_nips24_cvrp/catNips2024Code_0313/LEHD_main/CVRP/result/"
# my_model_path = path + data_source +"_" + method + str(sample_size) + "_" + str(my_epochs) + ".pt"






CUDA_DEVICE_NUM = 0
my_epochs = 40; problem_size = 0
# ================================
method = "org"
# ================================
sample_size = 128000
# ================================
RRC_budget = 500
# ================================
path = "/cat_nips24_cvrp/catNips2024Code_0313/LEHD_main/CVRP/result/"
my_model_path = path + data_source + "_" + str(my_epochs) + ".pt"



