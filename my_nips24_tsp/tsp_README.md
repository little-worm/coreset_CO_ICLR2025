# Coreset for TSP

Our code is base on [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver), [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO) and [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).


Working directory is set as `my_nips24_tsp\`.

## Environment configuration

Refer to `tsp_env.md` for environment configuration.

More details for environment configuration are in `tsp_environment.yml`.



## tsp data generation
- For generate artificial data (from uniform distribution and normal distribution):
`bash catNips2024Code_0313/myDIFUSCO/data/my_data_scripts/my_tsp_data.bash` 
`python catNips2024Code_0313/myDIFUSCO/data/generate_tsp_data_plus.py`
- For generate `TSPlib`  data:
`python catNips2024Code_0313/myDIFUSCO/data/my_tsplib_data.py` 



# coreset generation
- Generate RWD coreset for time efficiency: `python catNips2024Code_0313/_my_CO2024/myCoreset/my_generate_RWD_data_old.py`
- Generate RWD coreset : `python catNips2024Code_0313/_my_CO2024/myCoreset/my_generate_RWD_data.py`
- Generate US coreset
`python catNips2024Code_0313/_my_CO2024/myCoreset/my_generate_uniformSampling.py`

- Alignment test instance with tree structure: `python catNips2024Code_0313/_my_CO2024/myCoreset/my_align_test_data.py`


## training and test models

For training and testing models, run the shell scripts in `/cat_nips24_tsp/catNips2024Code_0313/myDIFUSCO/my_scripts`.
For example, 
- for training 
- -  `bash catNips2024Code_0313/myDIFUSCO/my_scripts/trainN01_tsp100_2d/train_tsp100_RWDcoreset_2d.sh` 
- - `bash catNips2024Code_0313/myDIFUSCO/my_scripts/trainN01_tsp100_2d/train_tsp100_UScoreset_2d.sh` 
- - `bash catNips2024Code_0313/myDIFUSCO/my_scripts/trainN01_tsp100_2d/train_tsp100_org.sh`

- for test 
- - `bash catNips2024Code_0313/myDIFUSCO/my_scripts/trainN01_tsp100_2d/test_tsp100_org.sh`
- - `bash /cat_nips24_tsp/catNips2024Code_0313/myDIFUSCO/my_scripts/trainN01_tsp100_2d/test_tsp100_RWDcoreset_2d.sh`
- - `bash /cat_nips24_tsp/catNips2024Code_0313/myDIFUSCO/my_scripts/trainN01_tsp100_2d/test_tsp100_UScoreset_2d.sh`



> Note: Some pathnames may need to be changed.
