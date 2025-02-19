export PYTHONPATH="$PWD:$PYTHONPATH"
# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"



maxIterRWD=5;maxPlloNmu=40

my_node_num=100; point_dim=3
export CUDA_VISIBLE_DEVICES=2; my_two_opt_iterations=0; my_test_node_num=1000
if [ $my_test_node_num=100 ]; then
    my_test_sample_num=1280
else
    my_test_sample_num=128
fi
#<<< RWD 
my_ckpt_path="my_result/tsp100_3ddim/models/train-tsp100_RWD4103/54rlmvvj/checkpoints/last.ckpt"
my_Guass_std=1;  tmp_data_path_r='0.085'; tmp_data_path_kk='4'; my_coreset_size='4103';  dim_mode="ddim"; plus_size=3000
#----------
my_ckpt_path="my_result/tsp100_3ddim/models/train-tsp100_RWD7960/ekkl44ey/checkpoints/last.ckpt"
my_Guass_std=1;  tmp_data_path_r='0.07'; tmp_data_path_kk='4'; my_coreset_size='7960';  dim_mode="ddim"; plus_size=3000
#----------
my_ckpt_path="my_result/tsp100_3ddim/models/train-tsp100_RWD12058/knrsa58s/checkpoints/last.ckpt"
my_Guass_std=1;  tmp_data_path_r='0.064'; tmp_data_path_kk='4'; my_coreset_size='12058';  dim_mode="ddim"; plus_size=3000
#--------------------------- 



tmp_batch_size=64
tmp_num_epochs=20



my_test_data="tsp"$my_test_node_num"_test_Guass_0_1_seed666_"$my_test_sample_num"_3ddim_aligned.txt"
# my_test_data="tsp"$my_test_node_num"_test_Guass_0_2_seed666_"$my_test_sample_num"_3ddim_aligned.txt"
# my_test_data="tsp"$my_test_node_num"_test_Guass_0_4_seed666_"$my_test_sample_num"_3ddim_aligned.txt"
# my_test_data="tsp"$my_test_node_num"_test_Guass_0_8_seed666_"$my_test_sample_num"_3ddim_aligned.txt"
# my_test_data="tsp"$my_test_node_num"_test_Uniform_seed666_"$my_test_sample_num"_3ddim_aligned.txt"

#>>>









python -u catNips2024Code_0313/myDIFUSCO/difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "test"$my_test_node_num"-tsp100_RWD-align"$my_coreset_size \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --my_data_path "my_dataCO/DIFCUSO_data/tsp100_"$point_dim$dim_mode"_plus"$plus_size"/myRWD_data/res_tsp100_train_Guass_0_"$my_Guass_std"_seed1234_128000_"$point_dim$dim_mode"plus_"$plus_size"_r"$tmp_data_path_r"_kk"$tmp_data_path_kk"_pool"$maxPlloNmu"IterRWD"$maxIterRWD"_"$my_coreset_size  \
    --training_split "tsp100_train_Guass_0_"$my_Guass_std"_seed1234_128000_"$point_dim$dim_mode"plus_"$plus_size"_myRWDCoreset" \
    --validation_split $my_test_data \
    --test_split $my_test_data \
  --batch_size $tmp_batch_size \
  --num_epochs $tmp_num_epochs \
  --validation_examples 0 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --my_result_path "my_result/tsp$my_node_num"_"$point_dim$dim_mode" \
  --project_name "tsp"$my_node_num"_"$point_dim$dim_mode \
  --fp16 \
  --two_opt_iterations  $my_two_opt_iterations \
  --tmp_data_path_r $tmp_data_path_r  --tmp_data_path_kk $tmp_data_path_kk  --my_coreset_size $my_coreset_size \
  --ckpt_path $my_ckpt_path





