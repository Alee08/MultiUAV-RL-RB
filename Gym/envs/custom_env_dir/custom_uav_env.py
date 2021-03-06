# MULTI_UAV GYM ENVIRONMENT 

import gym
import sys
import numpy as np
from decimal import Decimal
from configuration import Config
from settings.services_features import *
from load_and_save_data import *
import scenario_objects
import agent
import plotting
from gym import spaces, logger
import os
from temprl.temprl.wrapper import TemporalGoalWrapper
from gym.spaces import Dict, Discrete
load = Loader()
load.maps_data()
load.users_clusters()
load.maps_status()
plot = plotting.Plot()
#from iter import j
from configuration import j
from settings.hosp_scenario import hosp_features

conf = Config()
hosp_f = hosp_features()

class UAVEnv(gym.Env):

    def __init__(self):
        # Setting upper and lower bounds for actions values:FIXED_CLUSTERS_NUM
        upper_limits = np.array([val[0] for val in conf.LIMIT_VALUES_FOR_ACTION])
        lower_limits = np.array([val[1] for val in conf.LIMIT_VALUES_FOR_ACTION])

        '''o = idx_pos() #indice nuova pos start UAV conf.UAVS_POS[o]
        self.o = o'''

        self.action_set_min = conf.ACTION_SPACE_3D_MIN if conf.DIMENSION_2D==False else conf.ACTION_SPACE_2D_MIN
        if (conf.UNLIMITED_BATTERY==True):
            self.q_table_action_set = self.action_set_min
        else:
            if (self.action_set_min==conf.ACTION_SPACE_2D_MIN):
                self.q_table_action_set = conf.ACTION_SPACE_2D_TOTAL
                self.charging_set = conf.ACTION_SPACE_2D_WHILE_CHARGING
                self.come_home_set = conf.ACTION_SPACE_2D_COME_HOME
            else:
                self.q_table_action_set = conf.ACTION_SPACE_3D_TOTAL
                self.charging_set = conf.ACTION_SPACE_3D_WHILE_CHARGING
                self.come_home_set = conf.ACTION_SPACE_3D_COME_HOME
        self.action_space = spaces.Discrete(len(self.q_table_action_set))
        self.nb_actions = self.action_space.n

        if conf.HOSP_SCENARIO == False:
            self.observation_space = spaces.Box(low=0, high=max(conf.CELLS_ROWS, conf.CELLS_COLS, conf.FULL_BATTERY_LEVEL), shape=(conf.CELLS_ROWS, conf.CELLS_COLS, conf.ITERATIONS_PER_EPISODE), dtype=np.float32) if conf.DIMENSION_2D==False else spaces.Box(low=0, high=max(conf.CELLS_ROWS, conf.CELLS_COLS), shape=(conf.CELLS_ROWS, conf.CELLS_COLS), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Tuple(tuple(map(self.get_agent_dict_space, range(conf.N_UAVS))))

        self.state = None
        self.obs_points = load.obs_points
        self.points_matrix = load._points_matrix
        self.cs_points = load.cs_points
        self.eNB_point = load.enb_point
        self.cells_matrix = load.cells_matrix
        self.obs_cells = load.obs_cells
        self.max_uav_height = max([obs._z_coord for obs in self.obs_cells]) if conf.DIMENSION_2D==False else 0
        self.cs_cells = load.cs_cells
        self.hosp_points = load.hosp_points
        self.hosp_cells = load.hosp_cells



        # Set the CS position according to the desired resolution cells:
        #self.initial_uavs_pos = agent.Agent.setting_agents_pos(self.cs_cells) if conf.UNLIMITED_BATTERY==False else conf.UAVS_POS[j] #conf.UAVS_POS[j]
        if conf.HOSP_SCENARIO==True:
            self.initial_uavs_pos = conf.UAVS_POS[j]
        else:
            if conf.UNLIMITED_BATTERY==False:
                self.initial_uavs_pos = agent.Agent.setting_agents_pos(self.cs_cells)
            else:
                self.initial_uavs_pos = conf.UAVS_POS
        #if isinstance(self.initial_uavs_pos, list)==False:
        #    self.initial_uavs_pos = [self.initial_uavs_pos]

        self.agents = agent.Agent.initialize_agents(self.initial_uavs_pos, self.max_uav_height, self.action_set_min) # Agents initialization
        self.eNB_cells = load.enb_cells
        self.points_status_matrix = load.points_status_matrix
        self.cells_status_matrix = load.cells_status_matrix
        self.clusterer = load.initial_clusterer
        self.cluster_centroids = load.initial_centroids
        self.users_clusters = load.initial_usr_clusters
        self.clusters_radiuses = load.initial_clusters_radiuses
        initial_users = load.initial_users
        for user in initial_users:
            # Set the users coordinates according to the desired resolution cell:
            user._x_coord /= conf.CELL_RESOLUTION_PER_COL 
            user._y_coord /= conf.CELL_RESOLUTION_PER_ROW
        self.users = initial_users
        self.users_walk_steps = []
        self.k_steps_to_walk = 0
        self.uav_footprint = conf.ACTUAL_UAV_FOOTPRINT 
        self.n_users = len(self.users)
        self.discovered_users = []
        self.current_requested_bandwidth = 0 # --> bandwidth requested from all the users belonging to the current (considered) UAV footprint.
        self.all_users_in_all_foots = []
        self.n_active_users = 0
        self.n_tr_active = 0
        self.n_ec_active = 0
        self.n_dg_active = 0
        self.agents_paths = [[self.get_agent_pos(self.agents[uav])] for uav in range(conf.N_UAVS)]
        #self.agents_paths = [[0 for iteration in range(conf.ITERATIONS_PER_EPISODE)] for uav in range(conf.N_UAVS)]
        self.last_render = 0
        self.instant_to_render = 0
        self.last_commands = [None for uav in range(conf.N_UAVS)]
        self.priority_count = {}
        self.cordinate = {}
        self.idx = 0
        self.ciao = False


    def get_agent_dict_space(self, i):
        d = {
            "x": Discrete(conf.CELLS_ROWS),
            "y": Discrete(conf.CELLS_COLS),
            "beep": Discrete(2),
            "color": Discrete(conf.PRIORITY_NUM),
        }
        return Dict(d)




    def step_agent(self, agent, action):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # - 2D/3D cases;                                                                                    #
        # - UNLIMITED/LIMITED battery;                                                                      #
        # - Constat battery consumption wich includes both services and motions of the UAVs;                #
        # - All the users have the same priority (each of the is served) and and ask for the same service;  # 
        # - It is possible to set multi-service UAVs only with the following settings:                      #
        #       * 3D scenario;                                                                              #
        #       * Limited UAV bandwidth;                                                                    #
        #       * discrete and discontinuos users service request (i.e. conf.INF_REQUEST=False).                 #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        #print("ECCOLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

        #print("\n------ ACTION ------", action, "\n")

        info = ""
        #print("self.get_agent_pos(agent))self.get_agent_pos(agent))self.get_agent_pos(agent))self.get_agent_pos(agent))", self.get_agent_pos(agent))
        self.agents_paths[agent._uav_ID].append(self.get_agent_pos(agent))
        #self.agents_paths[agent._uav_ID].append(self.get_agent_pos(agent))

        if (conf.UAV_STANDARD_BEHAVIOUR==False):
            current_action = self.q_table_action_set[action]
            # With "move(..)" method, the agent can do also a beep on the current cell:
            agent_pos_, agent_is_off_map = agent.move(current_action, self.cells_matrix) # --> move_2D_unlimited_battery
            #pos = list(agent_pos_)
            #traj_j = [[0,2], [1,2], [2,2], [3,2]]
            #traj_j = [[0, 5], [1, 5], [1, 6], [2, 6], [2, 6], [2, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [6, 5], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [1, 8], [2, 8], [3, 8], [3, 7], [4, 7], [5, 7], [5, 7], [5, 7], [5, 7], [6, 7], [7, 7], [7, 6], [7, 5], [7, 4], [7, 3], [6, 3], [5, 3], [5, 2], [4, 2], [4, 2], [4, 2]]
            '''if trajectory == True:
                if pos in traj_j:
                    agent_is_on_traj = True
                else:
                    agent_is_on_traj = False'''
        else:
            current_action = conf.ACTION_SPACE_STANDARD_BEHAVIOUR.index(action)
            agent_pos_, agent_is_off_map = agent.move_standard_behaviour(action)
        if conf.HOSP_SCENARIO == True:
            self.last_commands[agent._uav_ID] = current_action # --> Controlla che uav_ID funzioni correttamente --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if ( ((action==conf.GO_TO_CS_2D_INDEX) or (action==conf.CHARGE_2D_INDEX)) or ((action==conf.GO_TO_CS_3D_INDEX) or (action==conf.CHARGE_3D_INDEX)) ):
            agent._users_in_footprint = []
            current_users_in_footprint = []
        else:
            self.current_requested_bandwidth = 0
            current_users_in_footprint, bandwidth_request_in_current_footprint = agent.users_in_uav_footprint(self.users, self.uav_footprint, self.discovered_users)
            if (conf.MULTI_SERVICE==False):
                agent._users_in_footprint = current_users_in_footprint
            else:
                self.current_requested_bandwidth = bandwidth_request_in_current_footprint

        # Compute the number of users which are served by the current UAV agent (useless at the moment, since it is not return by this method):
        n_served_users = agent.n_served_users_in_foot(agent._users_in_footprint) # --> This mainly performs a SIDE-EFFECT on the info 'served or not served' related to the users.

        # For the current iteration, add the users inside the footprint of the current UAV agent:  
        for user_per_agent_foot in current_users_in_footprint:
            self.all_users_in_all_foots.append(user_per_agent_foot) # --> SIDE-EFFECT on 'self.all_users_in_all_foots'

        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]

        if (conf.DIMENSION_2D==False):
            agent._z_coord = agent_pos_[2]

        if (conf.UNLIMITED_BATTERY==True):
            if (conf.HOSP_SCENARIO==False):
                reward = self.reward_function_1(agent._users_in_footprint)
                s_ = (agent_pos_)
            else:
                #reward = self.reward_function_HOSP()
                reward = 0
                color_idx = hosp_f.get_color_id(self.cells_matrix[agent._occupied_cell_coords[1]][agent._occupied_cell_coords[0]]._priority)
                s_ = (agent_pos_, agent.beep, color_idx)
                #print("_prioriiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiity", color_idx)

        else:
            if (conf.MULTI_SERVICE==False):
                reward = self.reward_function_2(agent._users_in_footprint, agent._battery_level, agent._required_battery_to_CS)
            else:
                if ( (conf.MULTI_SERVICE==True) and (conf.INF_REQUEST==False) ):
                    reward = self.reward_function_3(agent._users_in_footprint, agent._battery_level, agent._required_battery_to_CS, self.n_tr_active, self.n_ec_active, self.n_dg_active)
            if (conf.HOSP_SCENARIO==False):
                s_ = (agent_pos_, agent._battery_level)
            else:
                color_idx = hosp_f.get_color_id(self.cells_matrix[agent._occupied_cell_coords[0]][agent._occupied_cell_coords[1]]._priority)
                s_ = (agent_pos_, agent._battery_level, agent.beep, color_idx)



        # Force border constraint:
        if (conf.HOSP_SCENARIO==True):
            #reward += self.agent_is_on_traj(agent_is_on_traj)
            #print(reward, "custom_uav_env - agent_is_on_traj")
            reward += self.reward_off_map(agent_is_off_map)
            ##print(reward, "custom_uav_env - reward_off_map")
            reward += agent.do_beep(action, self.cells_matrix, self.priority_count, self.ciao)
            ##print(reward, "custom_uav_env - do_beep + reward_off_map")
            reward += -0.01
            ##print(reward, "custom_uav_env -0.01 x iteration + do_beep + reward_off_map")
            #obs, reward, done, info = TemporalGoalWrapper.step_agent(self, agent, action)

            #self.ciao = False


        done, info = self.is_terminal_state(agent)
        if done==True:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", done, info = self.is_terminal_state(agent))
        #Non viene mai usato questo
        #print(done)

        if (done):
            #print("weeeeeeeeee")
            #breakpoint()

            if (info=="IS CRASHED"):
                reward = 0.0

            else:
                reward = 0.0


        # Every time the action is undertaken by the 'first' UAV, then increse 'self.last_render':
        if (agent._uav_ID==0):
            self.last_render += 1

        return s_, reward, done, info

    def step(self, actions):

        #print("ECCOLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        #print(actions)
        obs = [0 for uav in range(conf.N_UAVS)]
        rewards = [0 for uav in range(conf.N_UAVS)]
        dones = [0 for uav in range(conf.N_UAVS)]
        infos = [0 for uav in range(conf.N_UAVS)]

        for uav in range(conf.N_UAVS):
            #self.agents_paths[uav].append(self.get_agent_pos(self.agents[uav]))

            ob, r, d, i = self.step_agent(self.agents[uav], actions[uav])
            obs[uav] = ob
            rewards[uav] = r
            dones[uav] = d
            infos[uav] = i

        #self.last_render += 1

        return obs, rewards, dones, infos

    def cost_reward(self, battery_level, needed_battery):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Cost function base on the the battery consumption.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        alpha_s = 0
        alpha_c = 0

        if (battery_level > conf.CRITICAL_BATTERY_LEVEL):
            alpha_s = 1
            alpha_c = 0
        elif ( (battery_level >= conf.CRITICAL_BATTERY_LEVEL_2) and (battery_level > needed_battery) ):
            alpha_s = 0.8
            alpha_c = 0.2
        elif ( (battery_level >= conf.CRITICAL_BATTERY_LEVEL_3) and (battery_level > needed_battery) ):
            alpha_s = 0.5
            alpha_c = 0.5
        elif ( (battery_level >= conf.CRITICAL_BATTERY_LEVEL_4) and (battery_level > needed_battery) ):
            alpha_s = 0.2
            alpha_c = 0.8
        elif (battery_level <= needed_battery):
            alpha_s = 0
            alpha_c = 1
            
        reward_for_cost = needed_battery/battery_level if battery_level != 0 else 1

        return reward_for_cost, alpha_s, alpha_c

    def discount_for_user_wait(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Possible discount factor based on the waiting time for a service of the users.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        n_discovered_users = len(self.discovered_users)
        all_wait_times = sum([wait_time_for_cur_user._info[3] for wait_time_for_cur_user in self.discovered_users])
        avg_wait_time_for_disc_users = all_wait_times/n_discovered_users if n_discovered_users!=0 else 0.0

        discount_factor = 0.0
        if (avg_wait_time_for_disc_users>conf.CRITICAL_WAITING_TIME_FOR_SERVICE):
            discount_factor = conf.CRITICAL_WAITING_TIME_FOR_SERVICE/conf.NORMALIZATION_FACTOR_WAITING_TIME_FOR_SERVIE if avg_wait_time_for_disc_users<=conf.NORMALIZATION_FACTOR_WAITING_TIME_FOR_SERVIE else 1.0
        else:
            discount_factor = 0.0

        return discount_factor

    def reward_function_HOSP(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Reward function which takes into account only the percentage of covered users.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        reward = 0

        return reward


    def reward_function_1(self, users_in_footprint):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Reward function which takes into account only the percentage of covered users.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        n_users_in_footprint = len(users_in_footprint)
        reward = n_users_in_footprint/(self.n_users/conf.N_UAVS)

        # Case in which a UAV is covering a number of user greater than the one (hypothetically) assigned to each UAV: 
        if (reward>1):
            reward = 1.0

        
        discount_for_wait_time = self.discount_for_user_wait()
        reward -= discount_for_wait_time
        if (reward<0.0):
            reward = 0.0

        return reward

    def reward_function_2(self, users_in_footprint, battery_level, needed_battery):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Reward function extended 'reward_function_1': it itakes into account also the battery level.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # There is no need to take into account the case in which the agent is going to CS, because when it is 'coming home'
        # the method 'Agent.users_in_uav_footprint' returns always 0 (the agent is only focused on going to CS withouth serving anyone else).

        reward_for_users = self.reward_function_1(users_in_footprint)

        alpha_s = 1
        alpha_c = 0
        if (needed_battery==None):
            
            reward = alpha_s*reward_for_users

        else:

            reward_for_cost, alpha_s, alpha_c = self.cost_reward(battery_level, needed_battery)
            reward = alpha_s*reward_for_users + alpha_c*reward_for_cost

        return reward

    def reward_function_3(self, users_in_footprint, battery_level, needed_battery, n_tr_active_users, n_ec_active_users, n_dg_active_users):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Reward function extended 'reward_function_2': it itakes into account also the multi-service request and a possiple discount factor for the users waiting time. battery level. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        reward_for_users = self.reward_function_1(users_in_footprint)

        served_users_asking_for_service = 0
        n_served_tr_users = 0
        n_served_ec_users = 0
        n_served_dg_users = 0
        for user in users_in_footprint:
            
            if (user._info[1]!=UsersServices.NO_SERVICE.value):
                
                if (user._info[1]==UsersServices.THROUGHPUT_REQUEST.value):
                    n_served_tr_users += 1
                elif (user._info[1]==UsersServices.EDGE_COMPUTING.value):
                    n_served_ec_users += 1
                elif (user._info[1]==UsersServices.DATA_GATHERING.value):
                    n_served_dg_users += 1
                
                served_users_asking_for_service += 1

        n_tr_served_perc = n_served_tr_users/n_tr_active_users if n_tr_active_users!=0 else 0
        n_ec_served_perc = n_served_ec_users/n_ec_active_users if n_ec_active_users!=0 else 0
        n_dg_served_perc = n_served_dg_users/n_dg_active_users if n_dg_active_users!=0 else 0

        alpha_u = 0.5
        alpha_tr = 0.2
        alpha_ec = 0.2
        alpha_dg = 0.1

        reward_for_services_and_users = alpha_u*reward_for_users + alpha_tr*n_tr_served_perc + alpha_ec*n_ec_served_perc + alpha_dg*n_dg_served_perc
        reward = reward_for_services_and_users

        if (needed_battery!=None):

            reward_for_cost, alpha_s, alpha_c = self.cost_reward(battery_level, needed_battery)
            reward = alpha_s*reward_for_services_and_users + alpha_c*reward_for_cost

        discount_for_wait_time = self.discount_for_user_wait()
        reward -= discount_for_wait_time
        if (reward<0.0):
            reward = 0.0
        
        return reward

    def reward_off_map(self, agent_is_off_map):

        if agent_is_off_map:
            return -2.0 # --> To select better --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        else:
            return 0.0




    def agent_is_on_traj(self, agent_is_on_traj):

        if agent_is_on_traj:
            return -1.0  # --> To select better --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        else:
            return 0.0

    '''def reward_do_beep(self, agent):

        if (action==BEEP):
            cell = agent._occupied_cell_coords
            # cell.beep() --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if (cell._priority!=None):
                if cell.color not in self.grid.color_count:
                    self.grid.color_count[cell.color] = 0
                self.grid.color_count[cell.color] += 1
            if cell.bip_count >= 2:
                reward = -1.0 # self.config.reward_duplicate_beep # --> To select better!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            reward = 0.0

        return reward'''


    def get_active_users(self): # --> It could perform just a simple SIDE-EFFECT with no return values (in this way will be avoided the redundancy) --> !!!!!!!!!!!!!!!!
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Retturn the number of active users (i.e., asking for a service) per each service. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        n_active_users = 0
        n_inactive_users = 0 
        tr_users = 0
        ec_users = 0
        dg_users = 0

        n_tr_served = 0
        n_ec_served = 0
        n_dg_served = 0        

        for user in self.users:
            if (user._info[1]==UsersServices.NO_SERVICE.value):
                n_inactive_users += 1
            elif (user._info[1]==UsersServices.THROUGHPUT_REQUEST.value):
                tr_users += 1
                if (user._info[0]==True):
                    n_tr_served += 1
            elif (user._info[1]==UsersServices.EDGE_COMPUTING.value):
                ec_users += 1
                if (user._info[0]==True):
                    n_ec_served += 1
            elif (user._info[1]==UsersServices.DATA_GATHERING.value):
                dg_users += 1
                if (user._info[0]==True):
                    n_dg_served += 1

        n_active_users = tr_users + ec_users + dg_users
        
        self.n_active_users = n_active_users
        self.n_tr_active = tr_users
        self.n_ec_active = ec_users
        self.n_dg_active = dg_users

        return n_active_users, tr_users, ec_users, dg_users, n_tr_served, n_ec_served, n_dg_served

    def set_action_set(self, agent):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Set the action space equal to one involving action for 2D case with or without limitation on battery. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if ( (agent._battery_level <= conf.CRITICAL_BATTERY_LEVEL) and (agent._charging == False) ):
            agent._action_set = conf.ACTION_SPACE_2D_COME_HOME if conf.DIMENSION_2D==True else conf.ACTION_SPACE_3D_COME_HOME 
        elif (agent._charging == True):
            agent._action_set = conf.ACTION_SPACE_2D_WHILE_CHARGING if conf.DIMENSION_2D==True else conf.ACTION_SPACE_3D_WHILE_CHARGING
        elif ( (agent._coming_home == False) and (agent._charging == False) and (agent._battery_level > conf.CRITICAL_BATTERY_LEVEL) ):
            agent._action_set = conf.ACTION_SPACE_2D_MIN if conf.DIMENSION_2D==True else conf.ACTION_SPACE_3D_MIN
            agent._path_to_the_closest_CS = []
            agent._current_pos_in_path_to_CS = -1
            agent._required_battery_to_CS = None

        self.action_space = spaces.Discrete(len(agent._action_set))

    def noisy_measure_or_not(self, values_to_warp):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Return the position values of a UAV which could be warped or not according to a 'noise probability'.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        warped_values = []
        coord_idx = 1

        for value in values_to_warp:
            noise_prob = np.random.rand()
            
            if noise_prob < 0.1:
                gaussian_noise = np.random.normal(loc=0, scale=1)
                warped_value = round(value + gaussian_noise)
                
                # X coordinate case:
                if (coord_idx==1):
                    if (warped_value>=conf.AREA_WIDTH):
                        warped_value = conf.AREA_WIDTH - 0.5
                    elif (warped_value<=0):
                        warped_value = 0.5
                    else:
                        warped_value += 0.5
                
                # Y coordinates case:
                elif (coord_idx==2):
                    if (warped_value>=conf.AREA_HEIGHT):
                        warped_value = conf.AREA_HEIGHT - 0.5
                    elif (warped_value<=0):
                        warped_value = 0.5
                    else:
                        warped_value += 0.5
                
                # Z coordinate case (no error on this measurement):
                elif (coord_idx==3):
                    warped_value = value # --> Z coordinate has a larger step, thus a gaussian noise could lead to a 'KeyError': for this reason Z coordinate will be not affected by noise
                '''
                # Z coordinate case:
                elif (coord_idx==3):
                    if (warped_value>conf.MAXIMUM_AREA_HEIGHT):
                        warped_value = conf.MAXIMUM_AREA_HEIGHT - 0.5
                    elif (warped_value<conf.MIN_UAV_HEIGHT):
                        warped_value = conf.MIN_UAV_HEIGHT + 0.5
                    else:
                        print("ERRORE QUI")
                        warped_value += (conf.UAV_Z_STEP + 0.5)
                        if (warped_value>=conf.MAXIMUM_AREA_HEIGHT):
                            warped_value = conf.MAXIMUM_AREA_HEIGHT - 0.5
                '''
            else:
                warped_value = value

            coord_idx += 1

            warped_values.append(warped_value)
        
        return tuple(warped_values)

    def is_terminal_state(self, agent):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Return (terminal_state, info): a state is terminal for a UAV only if it is either crashed or charging.  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if (conf.UNLIMITED_BATTERY==False):
            if ( (agent._battery_level <= 0) and (not agent.check_if_on_CS()) ):
                agent._battery_level = 0
                agent._crashed = True

                return True, "IS CRASHED"

            elif (agent._charging==True):
                agent._crashed = False

                return True, "IS CHARGING"

            else:
                agent._crashed = False

                return False, "IS WORKING"
        else:
            return False, "IS WORKING" # --> TO DEFINE BETTER --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def get_agent_pos(self, agent):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Return the coordinates ( (x,y) or (x,y,z) according to 2D or 3D case) of the 'agent'. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        x = agent._x_coord
        y = agent._y_coord
        z = agent._z_coord

        return (x, y) if conf.DIMENSION_2D==True else (x, y, z)

    def render(self, where_to_save=None, episode=None, how_often_render=None):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Used to save a .gif related to the environment animation for the current episode. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        self.instant_to_render += 1

        if ( (self.instant_to_render==how_often_render) or (how_often_render==None) ):
            
            if (where_to_save!=None):
                print("\nSaving animation for episode:", episode)

            plot.plt_map_views(obs_cells=self.obs_cells, cs_cells=self.cs_cells, enb_cells=self.eNB_cells, hosp_cells=self.hosp_cells,
                               points_status_matrix=self.points_status_matrix, cells_status_matrix=self.cells_status_matrix, users=self.users,
                               centroids=self.cluster_centroids, clusters_radiuses=self.clusters_radiuses, area_height=conf.AREA_HEIGHT,
                               area_width=conf.AREA_WIDTH, N_cells_row=conf.CELLS_ROWS, N_cells_col=conf.CELLS_COLS, agents_paths=self.agents_paths,
                               path_animation=True, where_to_save=where_to_save, episode=episode, last_render=self.last_render)

            self.instant_to_render = 0

        print("Animation saved")

        self.last_render = 0
        self.agents_paths = [[] for uav in range(conf.N_UAVS)]







    def reset_priority(self):
        for i in range(conf.CELLS_COLS):
            for j in range(conf.CELLS_ROWS):
                self.cells_matrix[j][i].bip_count = 0
        #print("ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo", self.cells_matrix[j][i].bip_count)
        '''for c in self.points_matrix:
            c.bip_count = 0'''
        self.priority_count = {}
        self.beep = False
        #self.ciao = True






    def reset_uavs(self, agent,):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Replace the 'agent' on an initial position randomly selected among the ones available for UAVs (it happens only when a crash occurs). #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        #Da decidere se fare due funzioni separate per batteria e reset della posizione ad ogni episodio

        if (agent._battery_level == 0) or conf.HOSP_SCENARIO == True:
            agent._battery_level = conf.FULL_BATTERY_LEVEL
            arise_pos_idx = np.random.choice(range(conf.N_UAVS))
            arise_pos = self.initial_uavs_pos[arise_pos_idx]


            print("MISSION_ID:", j, "START_POS:", arise_pos)

            # print(arise_pos, "arise_posarise_posarise_pos")

            agent._x_coord = arise_pos[0]  # +0.5
            agent._y_coord = arise_pos[1]  # +0.5
            agent._z_coord = arise_pos[2]  # +0.5

            agent._charging = False
            agent._coming_home = False



    def update_users_requests(self, users):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # If a user is not asking for a service, then randomly select a service for that user (he/she can also keep going to not request any service).  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        for user in users:
            # Update current user request only if the current user is not already requesting a service:
            if (user._info[1]==UsersServices.NO_SERVICE.value):
                
                type_of_service = scenario_objects.User.which_service()

                if (type_of_service == UsersServices.THROUGHPUT_REQUEST.value):
                    service_quantity = scenario_objects.User.bitrate_request()
                elif (type_of_service == UsersServices.EDGE_COMPUTING.value):
                    service_quantity = scenario_objects.User.edge_computing_request()
                elif (type_of_service == UsersServices.DATA_GATHERING.value):
                    service_quantity = scenario_objects.User.data_gathering()
                else:
                    service_quantity = 0

                requested_service_life = scenario_objects.User.needed_service_life(type_of_service) if type_of_service!=UsersServices.NO_SERVICE.value else 0

                user._info[1] = type_of_service
                user._info[2] = requested_service_life
                user._info[5] = service_quantity

    def move_users(self, current_iteration):
        # # # # # # # # # # # # # # # # 
        # Move the users on the map.  #
        # # # # # # # # # # # # # # # #

        for user_idx in range(self.n_users):
            # Move all the users at each iteration step only if the current iteration step is lower than the number of steps to move the users:
            if (current_iteration<self.k_steps_to_walk):
                self.users[user_idx]._x_coord = self.users_walk_steps[user_idx][current_iteration][0]/conf.CELL_RESOLUTION_PER_COL
                self.users[user_idx]._y_coord = self.users_walk_steps[user_idx][current_iteration][1]/conf.CELL_RESOLUTION_PER_ROW
                self.users[user_idx]._z_coord = self.users_walk_steps[user_idx][current_iteration][2]

    def compute_users_walk_steps(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Perform SIDE EFFECT on 'k_steps_to_walk', 'users_walk_steps', 'self.clusterer', 'self.users_clusters', 'self.cluster_centroids' and 'self.clusters_radiuses'. #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        min_steps = 2
        max_steps = 5
        k_steps = np.random.random_integers(min_steps, max_steps)
        self.k_steps_to_walk = k_steps
        
        # Users random walk:
        users_walks = scenario_objects.User.k_random_walk(self.users, k_steps)
        self.users_walk_steps = users_walks

        # New clusters detection:
        if (conf.FIXED_CLUSTERS_NUM>0):
            self.clusterer, self.users_clusters = scenario_objects.User.compute_clusterer(self.users) # --> If you use a fixed clusters number, then vary that number; if you use a variable number, then this method will return more values.
        else:
            optimal_clusterer, users_clusters, optimal_clusters_num, current_best_silhoutte_score = scenario_objects.User.compute_clusterer(self.users, fixed_clusters=False)
            self.clusterer = optimal_clusterer
        
        self.cluster_centroids = scenario_objects.User.actual_users_clusters_centroids(self.clusterer)
        self.clusters_radiuses = scenario_objects.User.actual_clusters_radiuses(self.cluster_centroids, self.users_clusters, conf.FIXED_CLUSTERS_NUM) # --> You can also use a variable clusters number in order to fit better inside clusters the users who have moved (in this case set 'fixed_cluster=False' in 'compute_clusterer(..)')