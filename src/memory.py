import numpy as np

class Memory:
    def __init__(self, state_size, max_mem_size, num_act):
        self.state_size = state_size
        self.mem_size = max_mem_size
        self.num_act = num_act
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, self.state_size), dtype=np.int32)
        self.new_state_memory = np.zeros((self.mem_size, self.state_size), dtype=np.int32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = {'reward_veh': np.zeros(self.mem_size, dtype=np.float32),
                              'reward_ped': np.zeros(self.mem_size, dtype=np.float32),
                              'reward': np.zeros(self.mem_size, dtype=np.float32)}

    def _store_transition(self, old_state, current_state, old_action, reward,  reward_veh, reward_ped):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = old_state
        self.new_state_memory[index] = current_state
        self.action_memory[index] = old_action
        self.reward_memory['reward'][index] = reward
        self.reward_memory['reward_veh'][index] = reward_veh
        self.reward_memory['reward_ped'][index] = reward_ped
        #print('sample:{}'.format((self.state_memory, self.new_state_memory, self.action_memory)))
        self.mem_cntr += 1


    def _size_now(self):
        """
        :return: current memory size

        """
        return len(self.action_memory)

    # def _show_mem_counter(self):
    #     return [self.mem_cntr]

    def _get_sample(self):
        sample_dict = {'state': self.state_memory, 'new_state': self.new_state_memory,
                       'action': self.action_memory, 'reward': self.reward_memory}
        return sample_dict


    def _get_counter(self):

        return self.mem_cntr
