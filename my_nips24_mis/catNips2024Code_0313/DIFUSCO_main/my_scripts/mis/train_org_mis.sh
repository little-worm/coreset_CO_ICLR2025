export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"
echo $PWD




my_num_epochs=20

python -u catNips2024Code_0313/DIFUSCO_main/difusco/train.py \
  --task "mis" \
  --wandb_logger_name "train-mis_er-org" \
  --diffusion_type "gaussian" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --my_data_path "my_dataCO/DIFCUSO_data/" \
  --training_split "mis_er/train_data_3000/*gpickle" \
  --training_split_label_dir "mis_er/train_data_3000/train_annotations/" \
  --validation_split "mis_er/test_data/er-90-100/*gpickle" \
  --test_split  "mis_er/test_data/er-90-100/*gpickle" \
  --batch_size 32 \
  --num_epochs $my_num_epochs \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --use_activation_checkpoint \
  --my_result_path "my_result/mis_er-org"$my_num_epochs \
  --project_name "mis" \




#my_dataCO/DIFCUSO_data/mis/er_test
#  --ckpt_path "/your/mis_sat_categorical/ckpt_path/last.ckpt" \

#--storage_path "catNips2024Code_0313/DIFUSCO_main/data/mis-benchmark-framework/my_dataCO/DIFCUSO_data/" \







