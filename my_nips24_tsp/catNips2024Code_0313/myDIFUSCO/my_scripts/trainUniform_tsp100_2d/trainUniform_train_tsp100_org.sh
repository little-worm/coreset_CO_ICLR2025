export PYTHONPATH="$PWD:$PYTHONPATH"
# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"





export CUDA_VISIBLE_DEVICES=0
#--------------------------------------------------
tmp_batch_size=64
tmp_num_epochs=20
my_two_opt_iterations=0
my_node_num=100;point_dim=2;dim_mode='d'



python -u catNips2024Code_0313/myDIFUSCO/difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "trainUniform_tsp100_Guass_org" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --my_data_path "my_dataCO/DIFCUSO_data/tsp100_2d_plus3000"  \
    --training_split "tsp100_train_Uniform_seed1234_128000_2d.txt" \
    --validation_split "tsp100_test_Uniform_seed666_1280_2d.txt" \
    --test_split "tsp100_test_Uniform_seed666_1280_2d.txt" \
  --batch_size $tmp_batch_size \
  --num_epochs $tmp_num_epochs \
  --validation_examples 64 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --my_result_path "my_result/trainUniform_tsp_org" \
  --project_name "trainUniform_tsp"$my_node_num"_"$point_dim$dim_mode \
  --fp16 \
  --two_opt_iterations $my_two_opt_iterations



# tsp100_200_train_Uniform_seed1234_1280_2d.txt