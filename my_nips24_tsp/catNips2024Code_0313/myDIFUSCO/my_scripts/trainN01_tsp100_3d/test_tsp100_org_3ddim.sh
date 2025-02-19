export PYTHONPATH="$PWD:$PYTHONPATH"
# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

#--------------------------------------------------
tmp_batch_size=64
tmp_num_epochs=20
tmp_val_check_interval=1000

my_node_num=100; point_dim=3; dim_mode='ddim'
export CUDA_VISIBLE_DEVICES=1; my_two_opt_iterations=0; my_test_node_num=1000
if [ $my_test_node_num=100 ]; then
    my_test_sample_num=1280
else
    my_test_sample_num=128
fi

#<<< RWD 226776
my_ckpt_path="my_result/tsp_org/models/train-tsp100_Guass_org/rizqra32/checkpoints/last.ckpt"
#>>>

my_test_data="tsp"$my_test_node_num"_3ddim_plus3000/tsp"$my_test_node_num"_test_Guass_0_1_seed666_"$my_test_sample_num"_3ddim.txt"
# my_test_data="tsp"$my_test_node_num"_3ddim_plus3000/tsp"$my_test_node_num"_test_Guass_0_2_seed666_"$my_test_sample_num"_3ddim.txt"
# my_test_data="tsp"$my_test_node_num"_3ddim_plus3000/tsp"$my_test_node_num"_test_Guass_0_4_seed666_"$my_test_sample_num"_3ddim.txt"
# my_test_data="tsp"$my_test_node_num"_3ddim_plus3000/tsp"$my_test_node_num"_test_Guass_0_8_seed666_"$my_test_sample_num"_3ddim.txt"
# my_test_data="tsp"$my_test_node_num"_3ddim_plus3000/tsp"$my_test_node_num"_test_Uniform_seed666_"$my_test_sample_num"_3ddim.txt"


python -u catNips2024Code_0313/myDIFUSCO/difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "test"$my_test_node_num"-tsp100_Guass_org" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --my_data_path "my_dataCO/DIFCUSO_data"  \
    --training_split "tsp100_3ddim_plus3000/tsp100_train_Guass_0_1_seed1234_128000_3ddimplus_3000.txt" \
    --validation_split $my_test_data \
    --test_split $my_test_data \
  --batch_size $tmp_batch_size \
  --num_epochs $tmp_num_epochs \
  --validation_examples 0 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --my_val_check_interval $tmp_val_check_interval \
  --my_result_path "my_result/tsp_org" \
  --project_name "tsp"$my_node_num"_"$point_dim$dim_mode \
  --fp16 \
  --two_opt_iterations $my_two_opt_iterations \
  --ckpt_path $my_ckpt_path





