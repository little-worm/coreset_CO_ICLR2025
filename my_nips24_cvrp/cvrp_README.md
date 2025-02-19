# Coreset for CVRP

Our code is base on [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver), [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO) and [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).


Working directory is set as `my_nips24_CVRP\`.

## Environment configuration

Refer to `cvrp_env.md` for environment configuration.

More details for environment configuration are in `cvrp_environment.yml`.



## cvrp data generation
- For generate artificial data:
`bash catNips2024Code_0313/LEHD_main/CVRP/data/my_cvrp_data.sh`
`python catNips2024Code_0313/LEHD_main/CVRP/data/my_transform_orgCVRP_to_line_dataset.py`



# coreset generation
- Generate RWD coreset : `python catNips2024Code_0313/_my_CO2024/myCoreset/my_generate_RWDdata_LEHDcvrp.py`
- Generate US coreset
`python catNips2024Code_0313/_my_CO2024/myCoreset/my_generate_uniformSampling_LEHDcvrp.py`



## training and test models

- for training 
- -  run  `train.py` in `floder catNips2024Code_0313/LEHD_main/CVRP``

- for test 
- - run `test.py` in folder `catNips2024Code_0313/LEHD_main/CVRP/` 
- - run `test_inCVRPlib.py` in folder `catNips2024Code_0313/LEHD_main/CVRP/`



> Note: Some pathnames may need to be changed.
