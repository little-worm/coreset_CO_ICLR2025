export PYTHONPATH="$PWD:$PYTHONPATH"
# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

my_node_num=100
point_dim=2
maxIterRWD=5;maxPlloNmu=40


export CUDA_VISIBLE_DEVICES=1

#<<< RWD 
tmp_data_path_r='0.83'; tmp_data_path_kk='4'; my_coreset_size='3973';  dim_mode="d"; plus_size=3000
# tmp_data_path_r='0.78'; tmp_data_path_kk='4'; my_coreset_size='8226';  dim_mode="d"; plus_size=3000
# tmp_data_path_r='0.752';  tmp_data_path_kk='4'; my_coreset_size='12235'; dim_mode="d"; plus_size=3000
# #---------------------------


my_two_opt_iterations=0
#--------------------------- 
tmp_batch_size=64
tmp_num_epochs=20
#my_test_data="tsp100_test_Guass_0_"$my_Guass_std"_seed666_1280_"$point_dim$dim_mode"_aligned.txt"
my_test_data="../../tsp100_test_Uniform_seed666_1280_2d.txt"

#>>>









python -u catNips2024Code_0313/myDIFUSCO/difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "trainUniform_-tsp100_RWD"$my_coreset_size \
  --diffusion_type "categorical" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --my_data_path "my_dataCO/DIFCUSO_data/tsp100_"$point_dim$dim_mode"_plus"$plus_size"/myRWD_data/res_tsp100_train_Uniform_seed1234_128000_"$point_dim$dim_mode"_r"$tmp_data_path_r"_kk"$tmp_data_path_kk"_pool"$maxPlloNmu"IterRWD"$maxIterRWD"_"$my_coreset_size  \
    --training_split "tsp100_train_Uniform_seed1234_128000_"$point_dim$dim_mode"_myRWDCoreset" \
    --validation_split $my_test_data \
    --test_split $my_test_data \
  --batch_size $tmp_batch_size \
  --num_epochs $tmp_num_epochs \
  --validation_examples 64 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --my_result_path "my_result/trainUniform_tsp$my_node_num"_"$point_dim$dim_mode" \
  --project_name "trainUniform_tsp"$my_node_num"_"$point_dim$dim_mode \
  --fp16 \
  --two_opt_iterations  $my_two_opt_iterations \
  --tmp_data_path_r $tmp_data_path_r  --tmp_data_path_kk $tmp_data_path_kk  --my_coreset_size $my_coreset_size





