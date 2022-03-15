import traci
import torch as T
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_EW_GREEN = 0  # action 0 code 00
PHASE_EWP_YELLOW = 1
PHASE_EW_YELLOW = 2
PHASE_NS_GREEN = 3  # action 1 code 01
PHASE_NSP_YELLOW = 4
PHASE_NS_YELLOW = 5
PHASE_EWV_GREEN = 6  # action 2 code 10
PHASE_EWV_YELLOW = 7
PHASE_NSV_GREEN = 8  # action 3 code 11
PHASE_NSV_YELLOW = 9
PHASE_TURN_GREEN = 10  # action 4 code 11
PHASE_TURN_YELLOW = 11
PHASE_P_GREEN = 12  # action 5 code 100
PHASE_P_YELLOW = 13
print('cuda on {}'.format(T.cuda.is_available()))


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, ped_green_duration,
                 yellow_duration, ped_yellow_duration, num_states, num_states_veh, num_actions, training_epochs, batch_size,
                 epsilon, epsilon_end, epsilon_dec, tau, max_mem_size):
        self.qnet_local = Model
        self.qnet_target = Model
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self.gamma = gamma
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._ped_green_duration = ped_green_duration
        self._yellow_duration = yellow_duration
        self._ped_yellow_duration = ped_yellow_duration
        self._num_state = num_states
        self._num_state_veh = num_states_veh
        self._Memory = Memory
        self._num_actions = num_actions
        self._training_epochs = training_epochs
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._epsilon_end = epsilon_end
        self._epsilon_dec = epsilon_dec
        self.tau = tau
        self._max_mem_size = max_mem_size
        self._reward_store = []
        self._veh_reward_store = []
        self._ped_reward_store = []

        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._avg_ped_queue_length_store = []


    def run(self, episode, epsilon):
        """
                Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self.stop = 0
        self._waiting_times = {}
        self._waiting_times_ped = {}
        self._sum_neg_reward = 0
        self._sum_neg_veh_reward = 0
        self._sum_neg_ped_reward = 0

        self._sum_queue_length = 0
        self._sum_ped_queue_length = 0

        self._sum_waiting_time = 0
        old_total_wait = 0
        old_total_wait_ped = 0
        old_state = -1
        old_action = -1


        while self._step < self._max_steps:
            # get current state of the intersection
            current_state_veh = self._get_state()
            current_state_ped = self._get_ped_state()
            current_state = current_state_veh + current_state_ped
            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            # In a certain interval, it is wise to make a current total waiting time
            current_total_wait_ped = self._collect_ped_waiting_times()
            reward_veh = old_total_wait - current_total_wait
            reward_ped = (old_total_wait_ped - current_total_wait_ped) / 50
            reward = reward_veh + reward_ped - self.stop*100
            # saving the data into the memory
            if self._step != 0:
                self._Memory._store_transition(old_state, current_state, old_action, reward, reward_veh, reward_ped)
            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                if old_action == 0 or old_action == 1:
                    self._set_yellow_phase(old_action + 1, act_bool=True)
                    self._simulate(self._ped_yellow_duration)
                    self._set_yellow_phase(old_action + 2, act_bool=True)
                    self._simulate(self._yellow_duration)
                elif old_action == 5:
                    self._set_yellow_phase(old_action)
                    self._simulate(self._ped_yellow_duration)
                else:
                    self._set_yellow_phase(old_action)
                    self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            if action == 5:
                self._simulate(self._ped_green_duration)
            else:
                self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            old_total_wait_ped = current_total_wait_ped

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward
            if reward_veh < 0:
                self._sum_neg_veh_reward += reward_veh
            if reward_ped < 0:
                self._sum_neg_ped_reward += reward_ped
        self._save_episode_stats()
        # print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(self._epsilon, 2))
        print("Total reward:", self._sum_neg_reward,
              "Total veh reward:", self._sum_neg_veh_reward,
              "Total ped reward:", self._sum_neg_ped_reward,
              "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._learn()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _get_state(self):
        num_states = self._num_state_veh
        state = np.zeros(num_states)
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 100 - lane_pos
            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 5:
                lane_cell = 0
            elif lane_pos < 10:
                lane_cell = 1
            elif lane_pos < 15:
                lane_cell = 2
            elif lane_pos < 20:
                lane_cell = 3
            elif lane_pos < 25:
                lane_cell = 4
            # interval 10m
            elif lane_pos < 35:
                lane_cell = 5
            elif lane_pos < 45:
                lane_cell = 6
            # interval 20m
            elif lane_pos < 65:
                lane_cell = 7
            elif lane_pos < 85:
                lane_cell = 8
            elif lane_pos < 100:
                lane_cell = 9
            else:
                lane_cell = -1

            # finding the lane where the car is located
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_1" or lane_id == "W2TL_2" or lane_id == "WW2TL_1" or lane_id == "WW2TL_2":
                # 西の上のレーングループ
                lane_group = 0
            elif lane_id == "W2TL_3" or lane_id == "WW2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_1" or lane_id == "N2TL_2" or lane_id == "NN2TL_1" or lane_id == "NN2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3" or lane_id == "NN2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_1" or lane_id == "E2TL_2" or lane_id == "EE2TL_1" or lane_id == "EE2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3" or lane_id == "EE2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_1" or lane_id == "S2TL_2" or lane_id == "SS2TL_1" or lane_id == "SS2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3" or lane_id == "SS2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"
        state = list(map(int, state))
        return state

    def _get_ped_state(self):
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

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        #print('green duration:{}'.format(steps_todo))
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step
        # どこかの道の青信号が終わるまで
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            # calc ped's accumulated waiting times
            self._collect_ped_waiting_times()
            # to get id that gives emergency stops
            stop_id = list(traci.simulation.getEmergencyStoppingVehiclesIDList())
            self.stop += len(stop_id)
            self._step += 1  # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length  # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds



    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL", "EE2TL", "NN2TL", "WW2TL", "SS2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            # incoming roadsに入っていない道なので、elseでもし、waiting_timesに含まれていた場合
            # すでに交差点を抜けたことになる
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _collect_ped_waiting_times(self):
        front_area_signals = [":TL_w0_0", ":TL_w1_0", ":TL_w2_0", ":TL_w3_0"]
        ped_list = traci.person.getIDList()
        for ped_id in ped_list:
            wait_time = traci.person.getWaitingTime(ped_id)
            area = traci.person.getLaneID(ped_id)
            if area in front_area_signals:
                if wait_time >= 0.1:
                    self._sum_ped_queue_length += 1
                try:
                    self._waiting_times_ped[ped_id] += wait_time
                except:
                    self._waiting_times_ped[ped_id] = wait_time
            else:
                if ped_id in self._waiting_times_ped:
                    del self._waiting_times_ped[ped_id]
        total_waiting_time = sum(self._waiting_times_ped.values())
        return total_waiting_time

    def _choose_action(self, states, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if np.random.random() < epsilon:
            # random action
            # np.random.randint(0, self._num_actions - 1)
            return np.random.choice(self._num_actions - 1)
        else:
            # the best action given the current state
            state = T.tensor([states]).to(self.qnet_local.device)
            actions = self.qnet_local.forward(state.float())
            action = T.argmax(actions).item()
            # print('Chosen Action is {}'.format(action))
            return action

    def _set_yellow_phase(self, old_action, act_bool=False):
        """
        Activate the correct yellow light combination in sumo
        """
        if not act_bool:
            yellow_phase_code = old_action * 2 + 3  # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
            traci.trafficlight.setPhase("C", yellow_phase_code)
        else:
            traci.trafficlight.setPhase("C", old_action)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        各方角に対して、緑信号をだす。10steps分
        """
        if action_number == 0:
            traci.trafficlight.setPhase("C", PHASE_EW_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("C", PHASE_NS_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("C", PHASE_EWV_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("C", PHASE_NSV_GREEN)
        elif action_number == 4:
            traci.trafficlight.setPhase("C", PHASE_P_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL") + traci.edge.getLastStepHaltingNumber("NN2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL") + traci.edge.getLastStepHaltingNumber("SS2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL") + traci.edge.getLastStepHaltingNumber("NN2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL") + traci.edge.getLastStepHaltingNumber("WW2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _learn(self):

        mem_cntr = self._Memory._get_counter()

        if mem_cntr < self._batch_size:
            return
        # Here is kinda fishy
        self.qnet_local.optimizer.zero_grad()
        max_mem = min(mem_cntr, self._max_mem_size)
        batch = np.random.choice(max_mem, self._batch_size, replace=False)

        batch_index = np.arange(self._batch_size, dtype=np.int32)

        sample = self._Memory._get_sample()
        state_batch = T.tensor(sample.get('state')[batch]).to(self.qnet_local.device)
        new_state_batch = T.tensor(sample.get('new_state')[batch]).to(self.qnet_local.device)
        reward_batch = T.tensor(sample.get('reward')['reward'][batch]).to(self.qnet_local.device)
        action_batch = sample.get('action')[batch]
        q_expected = self.qnet_local.forward(state_batch.float())[batch_index, action_batch]
        q_target_next = self.qnet_target.forward(new_state_batch.float())

        q_target = reward_batch + self.gamma * T.max(q_target_next, dim=1)[0]

        loss = self.qnet_local.loss(q_target, q_expected).to(self.qnet_local.device)
        loss.backward()
        self.qnet_local.optimizer.step()
        self.soft_update(self.qnet_local, self.qnet_target, self.tau)

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._veh_reward_store.append(self._sum_neg_veh_reward)
        self._ped_reward_store.append(self._sum_neg_ped_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        self._avg_ped_queue_length_store.append(self._sum_ped_queue_length / self._max_steps)

        result = {'reward': self._reward_store, 'reward_veh': self._veh_reward_store, 'reward_ped': self._ped_reward_store,
                  'cumulative_wait': self._cumulative_wait_store, 'avg_queue_len': self._avg_queue_length_store,
                  'avg_ped_queue_len': self._avg_ped_queue_length_store}
        return result
