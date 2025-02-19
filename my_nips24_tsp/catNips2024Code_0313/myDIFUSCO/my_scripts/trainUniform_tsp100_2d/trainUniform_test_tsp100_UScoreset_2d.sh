export PYTHONPATH="$PWD:$PYTHONPATH"
# shellcheck disable=SC2155
export WANDB_DISABLE_JOB_CREATION=true



my_node_num=100; point_dim=2
export CUDA_VISIBLE_DEVICES=2



#<<< 
my_ckpt_path="my_result/trainUniform_tsp100_2d/models/trainUniform_train-tsp100_US3973/lvwy83gl/checkpoints/last.ckpt"
tmp_data_path_r='0.83'; tmp_data_path_kk='4'; my_coreset_size='3973';  dim_mode="d"; plus_size=3000
#----------
# my_ckpt_path="my_result/trainUniform_tsp100_2d/models/trainUniform_train-tsp100_US8226/ppbiuaco/checkpoints/last.ckpt"
# tmp_data_path_r='0.78'; tmp_data_path_kk='4'; my_coreset_size='8226';  dim_mode="d"; plus_size=3000
# #----------
# my_ckpt_path="my_result/trainUniform_tsp100_2d/models/trainUniform_train-tsp100_US12235/245ily07/checkpoints/last.ckpt"
# tmp_data_path_r='0.752';  tmp_data_path_kk='4'; my_coreset_size='12235'; dim_mode="d"; plus_size=3000
# #--------------------------- 



tmp_batch_size=64
tmp_num_epochs=20


# my_test_data="../tsp"$my_test_node_num"_2d_plus3000/tsp"$my_test_node_num"_test_Guass_0_1_seed666_"$my_test_sample_num"_2d.txt"
# my_test_data="../tsp"$my_test_node_num"_2d_plus3000/tsp"$my_test_node_num"_test_Guass_0_2_seed666_"$my_test_sample_num"_2d.txt"
# my_test_data="../tsp"$my_test_node_num"_2d_plus3000/tsp"$my_test_node_num"_test_Guass_0_4_seed666_"$my_test_sample_num"_2d.txt"
# my_test_data="../tsp"$my_test_node_num"_2d_plus3000/tsp"$my_test_node_num"_test_Guass_0_8_seed666_"$my_test_sample_num"_2d.txt"
# my_test_data="../tsp"$my_test_node_num"_2d_plus3000/tsp"$my_test_node_num"_test_Uniform_seed666_"$my_test_sample_num"_2d.txt"

#>>>



  
     
for my_test_node_num in 100 200 500 1000; do
    if [[ "$my_test_node_num" -eq 100 ]]; then
      my_test_sample_num=1280
    else
      my_test_sample_num=128
    fi
    my_test_data_list=(
                    # "../tsp"$my_test_node_num"_"$point_dim$dim_mode"_plus"$plus_size"/tsp"$my_test_node_num"_test_Guass_0_1_seed666_"$my_test_sample_num"_2d.txt"
                    # "../tsp"$my_test_node_num"_"$point_dim$dim_mode"_plus"$plus_size"/tsp"$my_test_node_num"_test_Guass_0_2_seed666_"$my_test_sample_num"_2d.txt"
                    # "../tsp"$my_test_node_num"_"$point_dim$dim_mode"_plus"$plus_size"/tsp"$my_test_node_num"_test_Guass_0_4_seed666_"$my_test_sample_num"_2d.txt"
                    # "../tsp"$my_test_node_num"_"$point_dim$dim_mode"_plus"$plus_size"/tsp"$my_test_node_num"_test_Guass_0_8_seed666_"$my_test_sample_num"_2d.txt"
                    "../tsp"$my_test_node_num"_"$point_dim$dim_mode"_plus"$plus_size"/tsp"$my_test_node_num"_test_Uniform_seed666_"$my_test_sample_num"_2d.txt"
                    )
  for my_two_opt_iterations in 0 1000; do
    for my_test_data in "${my_test_data_list[@]}"; do

      export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
      echo "WANDB_ID is $WANDB_RUN_ID"



      python -u catNips2024Code_0313/myDIFUSCO/difusco/train.py \
        --task "tsp" \
        --wandb_logger_name "trainUniform_test"$my_test_node_num"-tsp100_US"$my_coreset_size \
        --diffusion_type "categorical" \
        --do_test \
        --learning_rate 0.0002 \
        --weight_decay 0.0001 \
        --lr_scheduler "cosine-decay" \
        --my_data_path "my_dataCO/DIFCUSO_data/tsp"$my_node_num"_"$point_dim$dim_mode"_plus"$plus_size  \
          --training_split "myUniformSample_data/tsp100_train_Uniform_seed1234_128000_"$point_dim$dim_mode"_"$my_coreset_size"_seed1234.txt" \
          --validation_split $my_test_data \
          --test_split $my_test_data \
        --batch_size $tmp_batch_size \
        --num_epochs $tmp_num_epochs \
        --validation_examples 0 \
        --inference_schedule "cosine" \
        --inference_diffusion_steps 50 \
        --my_result_path "my_result/trainUniform_tsp$my_node_num"_"$point_dim$dim_mode" \
        --project_name "trainUniform_tsp"$my_node_num"_"$point_dim$dim_mode \
        --fp16 \
        --two_opt_iterations $my_two_opt_iterations \
        --tmp_data_path_r $tmp_data_path_r  --tmp_data_path_kk $tmp_data_path_kk  --my_coreset_size $my_coreset_size \
        --ckpt_path $my_ckpt_path

    done
  done
done

