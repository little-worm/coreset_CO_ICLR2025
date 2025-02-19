export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
# shellcheck disable=SC2155




my_sample_size=6250; my_ckpt_path="my_result/mis_er-GWD6250/models/train-mis_er-GWD6250/sit8s637/checkpoints/last.ckpt"
my_sample_size=12417; my_ckpt_path="my_result/mis_er-GWD12417/models/train-mis_er-GWD12417/772ksrj1/checkpoints/last.ckpt"
# my_sample_size=19086; my_ckpt_path="my_result/mis_er-GWD19086/models/train-mis_er-GWD19086/174p3azm/checkpoints/last.ckpt"
my_test_split_list=(
                    # "mis_sat/test_data250_1065/*gpickle"
                    # "mis_sat/test_data200_860/*gpickle"
                    # "mis_sat/test_data150_645/*gpickle"
                    # "mis_sat/test_data403_10/*gpickle"
                    # "mis_sat/test_data403_30/*gpickle"
                    # "mis_sat/test_data403_50/*gpickle"
                    # "mis_sat/test_data403_70/*gpickle"
                    # "mis_sat/test_data403_90/*gpickle"    
                    "mis_sat/mis_test_sat/*gpickle"                
                  )
for my_parallel_sampling in 1 4; do
  for my_test_split in "${my_test_split_list[@]}"; do
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    echo "WANDB_ID is $WANDB_RUN_ID"
    echo $PWD


    python -u catNips2024Code_0313/DIFUSCO_main/difusco/train.py \
      --task "mis" \
      --wandb_logger_name "test-SATLIB-GWD"$my_sample_size"_"$my_parallel_sampling  \
      --diffusion_type "gaussian" \
      --do_test \
      --learning_rate 0.0002 \
      --weight_decay 0.0001 \
      --lr_scheduler "cosine-decay" \
      --my_data_path "my_dataCO/DIFCUSO_data/" \
      --training_split "mis_er/train_data_3000/*gpickle" \
      --training_split_label_dir "mis_er/train_data_3000/train_annotations/" \
      --validation_split $my_test_split \
      --test_split $my_test_split \
      --batch_size 32 \
      --num_epochs 20 \
      --validation_examples 0 \
      --inference_schedule "cosine" \
      --inference_diffusion_steps 50 \
      --parallel_sampling $my_parallel_sampling \
      --use_activation_checkpoint \
      --ckpt_path $my_ckpt_path \
      --my_result_path "my_result/SATLIB-GWD"$my_sample_size \
      --project_name "mis" \
      --resume_weight_only

  done
done













