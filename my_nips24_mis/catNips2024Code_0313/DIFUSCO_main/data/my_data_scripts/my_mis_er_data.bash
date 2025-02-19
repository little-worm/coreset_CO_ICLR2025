cd catNips2024Code_0313/DIFUSCO_main/data/mis-benchmark-framework || exit

python -u main.py gendata \
    random \
    None \
    ../../../../my_dataCO/DIFCUSO_data/mis_er/train_data_3000 \
    --model er \
    --min_n 90 \
    --max_n 100 \
    --num_graphs 1250 \
    --er_p 0.15






python -u main.py gendata \
    random \
    None \
    ../../../../my_dataCO/DIFCUSO_data/mis_er/train_data_3000 \
    --model er \
    --min_n 100 \
    --max_n 1000 \
    --num_graphs 300 \
    --er_p 0.15


#cd mis-benchmark-framework || exit

mkdir -p /tmp/gpus
touch /tmp/gpus/.lock
touch /tmp/gpus/0.gpu

python -u main.py \
    solve \
    kamis \
    ../../../../my_dataCO/DIFCUSO_data/mis_er/train_data_3000 \
    ../../../../my_dataCO/DIFCUSO_data/mis_er/train_data_3000/train_annotations \
    --time_limit 60    









# my_min_n=90; my_max_n=100


# python -u main.py gendata \
#     random \
#     None \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n \
#     --model er \
#     --min_n $my_min_n \
#     --max_n $my_max_n \
#     --num_graphs 128 \
#     --er_p 0.15


# #cd mis-benchmark-framework || exit

# mkdir -p /tmp/gpus
# touch /tmp/gpus/.lock
# touch /tmp/gpus/0.gpu

# python -u main.py \
#     solve \
#     kamis \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n/train_annotations \
#     --time_limit 60    







# my_min_n=400; my_max_n=500


# python -u main.py gendata \
#     random \
#     None \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n \
#     --model er \
#     --min_n $my_min_n \
#     --max_n $my_max_n \
#     --num_graphs 128 \
#     --er_p 0.15


# #cd mis-benchmark-framework || exit

# mkdir -p /tmp/gpus
# touch /tmp/gpus/.lock
# touch /tmp/gpus/0.gpu

# python -u main.py \
#     solve \
#     kamis \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n/train_annotations \
#     --time_limit 60    









# my_min_n=700; my_max_n=800


# python -u main.py gendata \
#     random \
#     None \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n \
#     --model er \
#     --min_n $my_min_n \
#     --max_n $my_max_n \
#     --num_graphs 128 \
#     --er_p 0.15


# #cd mis-benchmark-framework || exit

# mkdir -p /tmp/gpus
# touch /tmp/gpus/.lock
# touch /tmp/gpus/0.gpu

# python -u main.py \
#     solve \
#     kamis \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n \
#     ../../../../my_dataCO/DIFCUSO_data/mis_er/test_data/er-$my_min_n-$my_max_n/train_annotations \
#     --time_limit 60    





# 生成 125000 + 3000
# min90,max100 + min100,min1000