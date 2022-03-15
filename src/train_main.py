from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import datetime
import numpy as np
import torch as T

import warnings
warnings.simplefilter('ignore')

# FOR SIT PC
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# FOR my ubuntu laptop
# os.system("export SUMO_HOME=/usr/share/sumo")
# try:
#     sys.path.append("/usr/share/sumo/tools")
# except ImportError:
#     sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
from simulation import Simulation
from utils import import_train_configuration, set_sumo, set_train_path
from gen_vp import TrafficGenerator
# from dqn_net import DeepQNetwork
from ddqn_net import DeepQNetwork
from memory import Memory
from visual import Visualization

if __name__ == '__main__':
    config = import_train_configuration(config_file='sim.ini')
    sumo_cmd_f = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    print('config:{}'.format(config))
    print('sumo_cmd:{}'.format(sumo_cmd_f))

    TrafficGen = TrafficGenerator(
        max_steps=config['max_steps'],
        n_cars_generated=config['n_cars_generated'],
        n_peds_generated=config['n_peds_generated']
    )

    Memory = Memory(
        state_size=config['num_states'],
        max_mem_size=config['max_mem_size'],
        num_act=5
    )

    Model = DeepQNetwork(
        lr=config['lr'],
        input_dims=config['num_states'],
        target_input_dims=config['num_states'],
        fc1_dims=config['fc1_dims'],
        fc2_dims=config['fc2_dims'],
        fc3_dims=config['fc3_dims'],
        n_actions=config['num_actions']
    )
    Simulation = Simulation(
        Model=Model,
        Memory=Memory,
        TrafficGen=TrafficGen,
        sumo_cmd=sumo_cmd_f,
        gamma=config['gamma'],
        max_steps=config['max_steps'],
        green_duration=config['green_duration'],
        ped_green_duration=config['ped_green_duration'],
        yellow_duration=config['yellow_duration'],
        ped_yellow_duration=config['ped_yellow_duration'],
        num_states=config['num_states'],
        num_states_veh=config['num_state_veh'],
        num_actions=config['num_actions'],
        training_epochs=config['training_epochs'],
        batch_size=config['batch_size'],
        epsilon=config['epsilon'],
        epsilon_end=config['epsilon_end'],
        epsilon_dec=config['epsilon_dec'],
        tau=config['tau'],
        max_mem_size=config['max_mem_size']
    )

    Visualization = Visualization(
        path,
        dpi=96
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])
        simulation_time = Simulation.run(episode=episode, epsilon=epsilon)  # run the simulation

        print('Simulation time:', simulation_time, 's - Training time:', 's - Total:',
              np.round(simulation_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())

    # T.save(Model.state_dict(), os.path.join(path, 'trained_model.pth'))
    T.save(Model, os.path.join(path, 'trained_model.pth'))
    result_data = Simulation._save_episode_stats()
    Visualization.save_data_and_plotly_data(reward_data=result_data['reward'], x_rng=config['total_episodes'], reward_ped_data=None, filename='reward')
    Visualization.save_data_and_plotly_data(reward_data=result_data['reward_veh'], reward_ped_data=result_data['reward_ped'], x_rng=config['total_episodes'], filename='reward', multi=True)
    Visualization.save_data_and_plot(data=result_data['avg_queue_len'], filename='queue', xlabel='Episode',
                                     ylabel='Average queue length (vehicles)')
    Visualization.save_data_and_plot(data=result_data['avg_ped_queue_len'], filename='queue_ped', xlabel='Episode',
                                     ylabel='Average queue length (pedestrians)')
