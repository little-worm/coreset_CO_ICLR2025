##########################################################################################
# Machine Environment Config
import numpy as np
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
my_train_flag = 'isTrain'
my_CUDA_DEVICE_NUM = 1
my_full_datasize = 10000
my_epochs = 10
my_train_episodes_size = 1024
my_train_batch_size = 64
my_sampler_weights = np.ones(my_full_datasize)
my_train_data_path = "_my_dataCO/LEHD_data/tsp_data/train_TSP100_n100w.txt"
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
from utils.utils import create_logger, copy_all_src,my_set_result_folder

from TSPTrainer import TSPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    "data_path": my_train_data_path,
    "my_train_flag":my_train_flag,
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

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [3001,],
        'gamma': 0.1
    }
}

trainer_params = {
    'my_sampler_weights': my_sampler_weights,
    'use_cuda': USE_CUDA,
    'cuda_device_num': my_CUDA_DEVICE_NUM,
    'train_full_datasize': my_full_datasize,
    'epochs': my_epochs,
    'my_train_episodes_size': my_train_episodes_size,
    'train_batch_size': my_train_batch_size,
    'logging': {
        'model_save_interval': my_model_save_interval,
        'img_save_interval': my_img_save_interval,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.

    }
}

my_desc = 'train__tsp_n' + str(env_params['pomo_size']) + '_epochs' + str(trainer_params['epochs']) + '_eposides' + str(trainer_params['my_train_episodes_size'])  + '_batch' + str(trainer_params['train_batch_size'])

logger_params = {
    'log_file': {
        'my_log_dir': my_log_dir,
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

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['my_train_episodes_size'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, my_CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

#if __name__ == "__main__":
#    print("------------------")
#    main()


main()
