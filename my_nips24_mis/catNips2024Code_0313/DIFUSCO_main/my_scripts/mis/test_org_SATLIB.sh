export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
# shellcheck disable=SC2155




my_ckpt_path="my_result/mis_er-org20/models/train-mis_er-org/kzst3ce4/checkpoints/last.ckpt"


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
      --wandb_logger_name "test-SATLIB-org_"$my_parallel_sampling  \
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
      --my_result_path "my_result/SATLIB-org"$my_num_epochs \
      --project_name "mis" \
      --resume_weight_only
  done
done

# python -u catNips2024Code_0313/DIFUSCO_main/difusco/train.py \
#   --task "mis" \
#   --wandb_logger_name "train-mis_er-org" \
#   --diffusion_type "gaussian" \
#   --do_train \
#   --do_test \
#   --learning_rate 0.0002 \
#   --weight_decay 0.0001 \
#   --lr_scheduler "cosine-decay" \
#   --my_data_path "my_dataCO/DIFCUSO_data/" \
#   --training_split "mis_er/train_data_3000/*gpickle" \
#   --training_split_label_dir "mis_er/train_data_3000/train_annotations/" \
#   --validation_split "mis_er/test_data/er-90-100/*gpickle" \
#   --test_split  "mis_er/test_data/er-90-100/*gpickle" \
#   --batch_size 32 \
#   --num_epochs $my_num_epochs \
#   --validation_examples 8 \
#   --inference_schedule "cosine" \
#   --inference_diffusion_steps 50 \
#   --use_activation_checkpoint \
#   --my_result_path "my_result/mis_er-org"$my_num_epochs \
#   --project_name "mis" \

