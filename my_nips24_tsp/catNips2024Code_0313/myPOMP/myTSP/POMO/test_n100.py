##########################################################################################
# Machine Environment Config
import numpy as np
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
my_train_flag = 'isTest'
my_CUDA_DEVICE_NUM = 1
my_test_datasize = 1000
my_test_batch_size = 64
my_test_model_path = '_my_results/myResPOMO/tspRes/20240109_171540_train__tsp_n100__3000epoch'
my_test_model_epoch = 10
my_test_data_path = "_my_dataCO/LEHD_data/tsp_data/test_TSP100_n1w.txt"
my_model_save_interval = 10
my_img_save_interval = 10
my_log_dir = './_my_results/myResPOMO/tspRes/'

##########################################################################################
# Path Config

import os
import sys
cfd = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.abspath(cfd+'../../../../..')
os.chdir(DIR)
sys.path.insert(0, os.path.abspath(cfd+'../../..'))  # for problem_def



##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTester import TSPTester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    "data_path": my_test_data_path,
    "my_train_flag": my_train_flag
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': my_CUDA_DEVICE_NUM,
    'model_load': {
        'path': my_test_model_path,  # directory path of pre-trained model and log files saved.
        'epoch': my_test_model_epoch,  # epoch version of pre-trained model to laod.
    },
    'my_test_datasize': 100*1000,
    'test_batch_size': my_test_batch_size,
    'augmentation_enable': True,
    'aug_factor': -1,
    'aug_batch_size': 100,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

my_desc = 'test__tsp_n' + str(env_params['pomo_size']) + '_test_datasize' + str(tester_params['my_test_datasize'])  + '_batch' + str(tester_params['test_batch_size'])


logger_params = {
    'log_file': {
        'my_log_dir':  my_log_dir,
        'desc': my_desc,
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['my_test_datasize'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, my_CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

#if __name__ == "__main__":
#    main()

main()
