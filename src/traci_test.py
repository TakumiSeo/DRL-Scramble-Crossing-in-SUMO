from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import datetime
import numpy as np
import warnings
warnings.simplefilter('ignore')

# FOR SIT PC
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

# FOR my ubuntu laptop
os.system("export SUMO_HOME=/usr/share/sumo")
try:
    sys.path.append("/usr/share/sumo/tools")
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci
from utils import import_train_configuration, set_sumo
from gen_vp import TrafficGenerator


def run(sumo_cmd, max_steps):
    step = 0
    traci.start(sumo_cmd)
    while step < max_steps:
        traci.simulationStep()
        ids = traci.person.getIDList()
        print(_get_ped_state())
        for ped_id in ids:
            lane_pos = traci.person.getLanePosition(ped_id)
            lane_id = traci.person.getLaneID(ped_id)
            if 'TL_w' in lane_id:
                print('ped:{}, pos:{}'.format(ped_id, lane_pos))
        step += 1
    traci.close()
    sys.stdout.flush()

def _get_ped_state():
    ped_list = traci.person.getIDList()
    # |w_state| = 40
    w_state = np.zeros((4, 10))
    # |c_state| = 36
    c_state = np.zeros((4, 9))
    # |xc_state| = 30
    xc_state = np.zeros((2, 15))
    # |state| = 40+36+30 = 106
    for ped_id in ped_list:
        lane_pos = traci.person.getLanePosition(ped_id)
        lane_id = traci.person.getLaneID(ped_id)
        lane_cell = int(lane_pos // 2)
        lane_cell_wait = int(lane_pos // 1)
        if lane_cell_wait >= 10:
            lane_cell_wait = 9
        if 'TL_w' in lane_id:
            if 'TL_w0_0' in lane_id:
                lane_group = 0
            elif 'TL_w1_0' in lane_id:
                lane_group = 1
            elif 'TL_w2_0' in lane_id:
                lane_group = 2
            elif 'TL_w3_0' in lane_id:
                lane_group = 3
            else:
                lane_group = -1
            for lg in range(4):
                if lane_group == lg:
                    if 0 <= lane_cell_wait <= 10:
                        w_state[lg][lane_cell_wait] = 1
        if lane_id in ['TL_c1_0', 'TL_c3_0', 'TL_c4_0', 'TL_c5_0']:
            if 'TL_c1_0' in lane_id:
                lane_group_c = 0
            elif 'TL_c3_0' in lane_id:
                lane_group_c = 1
            elif 'TL_c4_0' in lane_id:
                lane_group_c = 2
            elif 'TL_c5_0' in lane_id:
                lane_group_c = 3
            else:
                lane_group_c = -1
            for clg in range(4):
                if lane_group_c == clg:
                    if 0 <= lane_cell <= 9:
                        c_state[clg][lane_cell] = 1

        # for scramble crossing
        if lane_id in ['TL_c0_0', 'TL_c2_0']:
            if 'TL_c0_0' in lane_id:
                lane_group_xc = 0
            elif 'TL_c2_0' in lane_id:
                lane_group_xc = 1
            else:
                lane_group_xc = -1
            for clg in range(2):
                if lane_group_xc == clg:
                    if 0 <= lane_cell <= 15:
                        xc_state[clg][lane_cell] = 1

    w_state = list(map(int, w_state.flatten()))
    c_state = list(map(int, c_state.flatten()))
    xc_state = list(map(int, xc_state.flatten()))
    ped_state = w_state + c_state + xc_state
    return ped_state

if __name__ == '__main__':
    config = import_train_configuration(config_file='sim.ini')
    sumo_cmd_f = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    print('config:{}'.format(config))
    print('sumo_cmd:{}'.format(sumo_cmd_f))

    run(sumo_cmd=sumo_cmd_f, max_steps=config['max_steps'])
