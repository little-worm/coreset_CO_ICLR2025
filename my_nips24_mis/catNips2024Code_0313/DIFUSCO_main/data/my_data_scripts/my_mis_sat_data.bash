cd catNips2024Code_0313/DIFUSCO_main/data/mis-benchmark-framework || exit


my_min_n=403;my_max_n=10
# my_min_n=403;my_max_n=30
# my_min_n=403;my_max_n=50
# my_min_n=403;my_max_n=70
# my_min_n=403;my_max_n=90

# my_min_n=50;my_max_n=218
# my_min_n=100;my_max_n=430
my_min_n=150;my_max_n=645
# my_min_n=200;my_max_n=860
# my_min_n=250;my_max_n=1065


#    my_cnf_file/CBS_k3_n100_m$my_min_n"_b"$my_max_n \

python -u main.py gendata \
    sat \
    my_cnf_file/uf$my_min_n"-"$my_max_n \
    ../../../../my_dataCO/DIFCUSO_data/mis_sat/test_data$my_min_n"_"$my_max_n \
    --model er \
    --min_n $my_min_n \
    --max_n $my_max_n \
    --num_graphs 128 \
    --er_p 0.15


#cd mis-benchmark-framework || exit

mkdir -p /tmp/gpus
touch /tmp/gpus/.lock
touch /tmp/gpus/0.gpu

python -u main.py \
    solve \
    kamis \
    ../../../../my_dataCO/DIFCUSO_data/mis_sat/test_data$my_min_n"_"$my_max_n \
    ../../../../my_dataCO/DIFCUSO_data/mis_sat/test_data$my_min_n"_"$my_max_n/train_annotations \
    --time_limit 60    





