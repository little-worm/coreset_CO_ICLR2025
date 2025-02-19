# Coreset for MIS

Our code is base on [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver), [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO) and [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).


Working directory is set as `my_nips24_mis\`.

## Environment configuration

Details for environment configuration are in `mis_environment.yml`.



## mis data generation
- For generate artificial data (from uniform distribution and normal distribution):
`bash catNips2024Code_0313/DIFUSCO_main/data/my_data_scripts/my_mis_er_data.bash` 
`bash catNips2024Code_0313/DIFUSCO_main/data/my_data_scripts/my_mis_er_data_test.bash`


# coreset generation
- Generate RWD coreset : run `my_generate_GWDdata_MDS.py` in `catNips2024Code_0313/_my_CO2024/myCoreset_mis/`  
- Generate US coreset : run `my_generate_uniformSampling.py` in `catNips2024Code_0313/_my_CO2024/myCoreset_mis/` 



## training and test models

For training and testing models, run the shell scripts in `catNips2024Code_0313/myDIFUSCO/my_scripts`.
For example, 
- for training 
- -  `bash catNips2024Code_0313/DIFUSCO_main/my_scripts/mis/train_GWD_mis.sh` 
- - `bash catNips2024Code_0313/DIFUSCO_main/my_scripts/mis/train_US_mis.sh` 
- - `bash catNips2024Code_0313/DIFUSCO_main/my_scripts/mis/train_org_mis.sh`

- for test 
- - `bash catNips2024Code_0313/DIFUSCO_main/my_scripts/mis/test_GWD_mis.sh`
- - `bash catNips2024Code_0313/DIFUSCO_main/my_scripts/mis/test_US_mis.sh`
- - `bash catNips2024Code_0313/DIFUSCO_main/my_scripts/mis/test_org_mis.sh`



> Note: Some pathnames may need to be changed.




