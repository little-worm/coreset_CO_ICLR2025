import os

# data_source = "cvrp_0_100000_128t"
# data_source = "cvrp_0_025_125t"
data_source = "cvrp_0_010_125t"


## training----------------------------------------------------------
# 128000 baseline 
CUDA_DEVICE_NUM = 0
sample_size = 128000 
my_train_episodes = sample_size
my_epochs = 40; problem_size = 100; RRC_budget = 0; method = "org"
# =========================================
training_data_path = "/cat_nips24_cvrp/my_dataCO/LEHD_data/" + data_source + ".txt"
my_save_model_path = "/cat_nips24_cvrp/catNips2024Code_0313/LEHD_main/CVRP/result/" + data_source + "_" + str(my_epochs) + ".pt"
# =========================================






# # # # # RWD sampling
# CUDA_DEVICE_NUM = 3
# # ==========================================
# # 4697, 7694, 13908
# problem_size = 100; RRC_budget = 0
# # # sample_size = 13908; r = 17 # sample_size = 11428; r = 17.1 # sample_size = 8538; r = 17.9 # sample_size = 5885; r = 18.9  # sample_size = 12318; r = 17 # sample_size = 12551; r = 17 # sample_size = 3614; r = 19.6 # sample_size = 4314; r = 19.5 # sample_size = 3444; r = 19.7 # sample_size = 11755; r = 17.5
# # sample_size = 7694; r = 18
# # sample_size = 4697; r = 19
# # sample_size = 12033; r = 17.5
# #-------------------------------------------
# # sample_size = 4437; r = 45
# # sample_size = 8082; r = 40
# sample_size = 12175; r = 36.6
# # ==========================================
# # 4560, 9246,14447
# problem_size = 100; RRC_budget = 0
# # sample_size = 4560; r = 24
# # sample_size = 9246; r = 22
# # sample_size = 14447; r = 21
# # =========================================
# my_train_episodes = sample_size; my_epochs = 40
# method = "RWD"
# training_data_path = "/cat_nips24_cvrp/my_dataCO/LEHD_data/" + "my" + method + "_data/" + \
#                     "res_" + data_source + "_r" + str(r) + "_kk4_pool64IterRWD5_" + str(sample_size) + "/org/" + data_source + "_myRWDCoreset"
# my_save_model_path = "/cat_nips24_cvrp/catNips2024Code_0313/LEHD_main/CVRP/result/" + data_source + "_" + method + str(sample_size) + "_" + str(my_epochs) + ".pt"





# # # US sampling 
# CUDA_DEVICE_NUM = 2
# # ==========================================
# problem_size = 100; RRC_budget = 0
# # sample_size = 13908; r = 17 # sample_size = 11428; r = 17.1 # sample_size = 8538; r = 17.9 # sample_size = 5885; r = 18.9  # sample_size = 12318; r = 17 # sample_size = 12551; r = 17 # sample_size = 3614; r = 19.6 # sample_size = 4314; r = 19.5 # sample_size = 3444; r = 19.7 # sample_size = 11755; r = 17.5
# # sample_size = 7694; r = 18
# # sample_size = 4697; r = 19
# # sample_size = 12033; r = 17.5
# #-------------------------------------------
# # sample_size = 4437; r = 45
# sample_size = 8082; r = 40
# sample_size = 12175; r = 36.6
# # ==========================================
# my_train_episodes = sample_size; my_epochs = 40
# method = "US"
# training_data_path = "/cat_nips24_cvrp/my_dataCO/LEHD_data/" + "my" + method + "_data/" + \
#                     data_source + "_" + str(sample_size) + "_seed1234.txt"
# my_save_model_path = "/cat_nips24_cvrp/catNips2024Code_0313/LEHD_main/CVRP/result/" + data_source + "_" + method + str(sample_size) + "_" + str(my_epochs) + ".pt"





