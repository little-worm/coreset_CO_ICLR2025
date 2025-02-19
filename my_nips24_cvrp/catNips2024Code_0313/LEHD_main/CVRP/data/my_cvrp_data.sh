my_mean=0
my_std=100000



# python catNips2024Code_0313/LEHD_main/data/my_gen_cvrp_org_dataset.py \
python catNips2024Code_0313/LEHD_main/CVRP/data/my_gen_cvrp_org_dataset.py \
       --num_instances 10 \
       --num_nodes 100 \
       --capacity 50 \
       --working_dir "catNips2024Code_0313/LEHD_main/CVRP/data/cvrp_"$my_mean"_"$my_std/ \
       --seed 123 \
       --my_mean $my_mean --my_std $my_std



