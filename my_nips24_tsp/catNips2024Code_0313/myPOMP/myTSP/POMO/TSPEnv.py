
from dataclasses import dataclass
import torch,os
from tqdm import tqdm
import numpy as np
#from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    


class TSPEnv:
    def __init__(self, env_params, trainer_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.trainer_params = trainer_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)
        

        # &&&&&&&&&& LEHD &&&&&&&&&
        self.data_path = os.path.abspath(env_params['data_path'])
        self.my_episode_data_nodes = []
        self.my_episode_data_tours = []
        if self.env_params['my_train_flag'] == 'isTrain':
            self.my_sampler_weights = trainer_params['my_sampler_weights']
            self.my_train_episodes_size = trainer_params['my_train_episodes_size']
            ## my &&&&&&&&&&&&&&
            self.my_full_datasize = trainer_params['train_full_datasize']
        if self.env_params['my_train_flag'] == 'isTest':
            self.my_full_datasize = trainer_params['my_test_datasize']    
        self.my_full_data_nodes = []
        self.my_full_data_tours = []

    def load_problems(self,episode, batch_size, aug_factor=1):
        self.episode = episode
        self.batch_size = batch_size
        self.problems, self.solution = self.my_episode_data_nodes[episode:episode + batch_size], self.my_episode_data_tours[episode:episode + batch_size]
        #print("=====self.problems = ",self.problems)
        self.problem_size = self.problems.shape[1]

#        self.problems = get_random_problems(batch_size, self.problem_size)
#        # problems.shape: (batch, problem, 2)
#        if aug_factor > 1:
#            if aug_factor == 8:
#                self.batch_size = self.batch_size * 8
#                self.problems = augment_xy_data_by_8_fold(self.problems)
#                # shape: (8*batch, problem, 2)
#            else:
#                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances






    #####  LEHD &&&&&&&&&&&&&&&&&&&&
    
    def my_load_full_data(self,begin_index=0):

        print('load my-full-dataset begin!')

        self.my_full_data_nodes = []
        self.my_full_data_tours = []
        my_full_datasize = self.my_full_datasize
        for line in tqdm(open(self.data_path, "r").readlines()[0+begin_index : my_full_datasize+begin_index], ascii=True):
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

            self.my_full_data_nodes.append(nodes)
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

            self.my_full_data_tours.append(tour_nodes)





    def my_sampler_eposide_data(self):
        if self.env_params['my_train_flag'] == 'isTrain':
            assert self.my_full_datasize == len( list(self.my_sampler_weights) )
            print('sample my eposide data begin!')
            sample_weights = self.my_sampler_weights; sample_weights = sample_weights / sum(sample_weights)
            sampled_indexes = np.random.choice([i for i in range(self.my_full_datasize)], size=self.my_train_episodes_size, p=sample_weights)
            my_episode_data_nodes = np.array([ self.my_full_data_nodes[index]  for index in sampled_indexes ])
            my_episode_data_tours = np.array([ self.my_full_data_tours[index]  for index in sampled_indexes ])
            self.my_episode_data_nodes = torch.tensor(my_episode_data_nodes,requires_grad=False)   
            self.my_episode_data_tours = torch.tensor(my_episode_data_tours,requires_grad=False)   
            print(f'load my-episode-dataset done!', )
        if self.env_params['my_train_flag'] == 'isTest':
            self.my_episode_data_nodes = torch.tensor(self.my_full_data_nodes)
            self.my_episode_data_tours = torch.tensor(self.my_full_data_tours)
#   val= torch.tensor([item.cpu().detach().numpy() for item in val]).cuda()

