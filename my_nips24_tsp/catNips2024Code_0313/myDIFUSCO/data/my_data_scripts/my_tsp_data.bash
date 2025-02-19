my_sample_type="Guass"  my_Guass_mean=0  my_Guass_std=1
# my_sample_type="Uniform"
my_node_num_min=1000; my_node_num_max=$my_node_num_min
my_node_num=$my_node_num_min
my_batch_size=16
plus_size=3000


# #================================= 2D ================================

# #<<< Test data
# my_seed=666;    point_dim=2;    dim_mode="d"
# my_num_samples=128
# if [ $my_node_num_min == $my_node_num_max ]; then
# my_save_filename="tsp"$my_node_num_min"_test_"$my_sample_type
# else
# my_save_filename="tsp"$my_node_num_min"_"$my_node_num_max"_test_"$my_sample_type
# fi
#>>>



# #<<< Training data
# my_seed=1234;   point_dim=2;    dim_mode="d"
# my_num_samples=128
# if [ $my_node_num_min == $my_node_num_max ]; then
# my_save_filename="tsp"$my_node_num_min"_train_"$my_sample_type
# else
# my_save_filename="tsp"$my_node_num_min"_"$my_node_num_max"_train_"$my_sample_type
# fi
# #>>>



#================================= 3D ================================

#   #<<< Training data 
#   my_seed=1234; point_dim=3; dim_mode="ddim"
#   my_num_samples=128000
#   my_save_filename="tsp"$my_node_num"_train_"$my_sample_type
#   #>>>
#   
#<<< Test data
point_dim=3; dim_mode="ddim"
my_seed=666
my_num_samples=128
my_save_filename="tsp"$my_node_num"_test_"$my_sample_type
#>>>


 




if [ "$my_sample_type" == "Guass" ]; then
    my_save_filename=$my_save_filename"_"$my_Guass_mean"_"$my_Guass_std
fi

if [ "$my_sample_type" == "Uniform" ]; then
    my_save_filename=$my_save_filename
fi

my_save_filename=$my_save_filename"_seed"$my_seed"_"$my_num_samples"_"$point_dim$dim_mode".txt"

python -u catNips2024Code_0313/myDIFUSCO/data/generate_tsp_data.py  \
    --min_nodes $my_node_num_min  --max_nodes $my_node_num_max  \
    --num_samples $my_num_samples  --batch_size $my_batch_size  \
    --filename "my_dataCO/DIFCUSO_data/tsp$my_node_num"_"$point_dim$dim_mode"_plus"$plus_size/"$my_save_filename \
    --my_sample_type $my_sample_type \
    --seed $my_seed --Guass_mean $my_Guass_mean --Guass_std $my_Guass_std \
    --my_point_dim $point_dim --my_dim_mode $dim_mode
