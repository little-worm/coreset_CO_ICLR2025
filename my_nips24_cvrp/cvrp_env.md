```
pip install pytz
pip install matplotlib
pip install tqdm
pip install ipykernel
``` 



``` 20240114 DIFCUSO
conda remove --name cat --all 
conda create --name cat python=3.9
conda activate cat
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric==2.3.1
pip install pytorch-lightning==1.9.5
pip install wandb
pip install seaborn
pip install Cython
pip install pickle5
pip install POT
pip install tsplib95



pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch_geometric
pip install pytorch-lightning==1.9.5
pip install wandb
pip install seaborn
pip install Cython
pip install pickle5
pip install POT
pip install tsplib95

pip install vrplib


>>> cat  install lkh >>> 
wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz
tar xvfz LKH-3.0.6.tgz
cd LKH-3.0.6
make
<<<  cat install lkh <<< 

git clone https://github.com/jvkersch/pyconcorde
cd pyconcorde
pip install -e .


cd difusco/utils/cython_merge
python setup.py build_ext --inplace
cd -

```
We need to change `lkh_path = 'XXXX/LKH-3.0.6/LKH'` in `DIFCUSO/generate_tsp_data.py`.




>>>error>>>
wandb.sdk.service.service.ServiceStartProcessError: The wandb service process exited with 1. Ensure that `sys.executable` is a valid python interpreter. You can override it with the `_executable` setting or with the `WANDB__EXECUTABLE` environment variable.

pip install -U click

<<<error<<< 


>>>error>>>
ModuleNotFoundError: No module named 'lkh'
pip install lkh

<<<error<<< 





















