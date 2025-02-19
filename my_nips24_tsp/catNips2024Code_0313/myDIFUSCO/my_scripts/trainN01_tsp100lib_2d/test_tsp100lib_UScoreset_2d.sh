export PYTHONPATH="$PWD:$PYTHONPATH"
# shellcheck disable=SC2155
export WANDB_DISABLE_JOB_CREATION=true
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"


my_node_num=100; point_dim=2
export CUDA_VISIBLE_DEVICES=0; my_two_opt_iterations=1000; my_test_tsplib_node_num=1000


#<<< 
my_ckpt_path="my_result/tsp100_2d/models/train-tsp100_US4003/3es8y568/checkpoints/last.ckpt"
my_Guass_std=1;  tmp_data_path_r='0.065'; tmp_data_path_kk='4'; my_coreset_size='4003';  dim_mode="d"; plus_size=3000
#----------
# my_ckpt_path="my_result/tsp100_2d/models/train-tsp100_US8245/7tnukmrd/checkpoints/last.ckpt"
# my_Guass_std=1;  tmp_data_path_r='0.054'; tmp_data_path_kk='4'; my_coreset_size='8245';  dim_mode="d"; plus_size=3000
# #----------
# my_ckpt_path="my_result/tsp100_2d/models/train-tsp100_US12951/2cjj2maz/checkpoints/last.ckpt"
# my_Guass_std=1;  tmp_data_path_r='0.05'; tmp_data_path_kk='4'; my_coreset_size='12951';  dim_mode="d"; plus_size=3000

# # #---------------------------



tmp_batch_size=64
tmp_num_epochs=20

my_test_data="../tsplib/my_tsblib_Eud2d_threshod100-"$my_test_tsplib_node_num".txt"

#>>>



  
     



python -u catNips2024Code_0313/myDIFUSCO/difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "test-tsplib"$my_test_tsplib_node_num"-tsp100_US"$my_coreset_size \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --my_data_path "my_dataCO/DIFCUSO_data/tsp"$my_node_num"_"$point_dim$dim_mode"_plus"$plus_size  \
  --training_split "myUniformSample_data/tsp100_train_Guass_0_"$my_Guass_std"_seed1234_128000_"$point_dim$dim_mode"plus_"$plus_size"_"$my_coreset_size"_seed1234.txt" \
    --validation_split $my_test_data \
    --test_split $my_test_data \
  --batch_size $tmp_batch_size \
  --num_epochs $tmp_num_epochs \
  --validation_examples 0 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --my_result_path "my_result/tsp$my_node_num"_"$point_dim$dim_mode" \
  --project_name "tsplib"$my_node_num"_"$point_dim$dim_mode \
  --fp16 \
  --two_opt_iterations $my_two_opt_iterations \
  --tmp_data_path_r $tmp_data_path_r  --tmp_data_path_kk $tmp_data_path_kk  --my_coreset_size $my_coreset_size \
  --ckpt_path $my_ckpt_path


#   tmp_batch_size=64

  #  --validation_examples 64 \
