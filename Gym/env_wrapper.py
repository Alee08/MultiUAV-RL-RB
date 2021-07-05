import numpy as np
import sys
import pickle
from os import mkdir
from os.path import join, isdir
from numpy import linalg as LA
from math import sqrt, inf
from decimal import Decimal
import time
import gym
import envs
from gym import spaces, logger
from scenario_objects import Point, Cell, User, Environment
import plotting
from configuration import Config
from configuration import j
from settings.dir_gen import *
import agent
from Astar import *
from settings.hosp_scenario import hosp_features

conf = Config()

if (conf.HOSP_SCENARIO == True):
    from restraining_bolt import RestrainingBolt
from temp_wrapper import UAVsTemporalWrapper
from temprl.temprl.wrapper import *
#from iter import j

hosp_f = hosp_features()
# _________ Main training parameters:_________
if conf.HOSP_SCENARIO == True:
    SHOW_EVERY = 50
    LEARNING_RATE = 0.01         #alpha
    #DISCOUNT = 0.95 originale
    DISCOUNT = 0.99           #gamma
    EPSILON = 0.01
    EPSILON_DECR = False
    EPSILON_DECREMENT = 0.999
    #EPSILON_MIN = 0.01
    #EPSILON_MIN2 = 0.4 originale
    #EPSILON_MIN2 = 0.1
    EPSILON_MIN = 0.01
    EPSILON_MIN2 = 0.1
    max_value_for_Rmax = 100
    #ITERATIONS_PER_EPISODE = 160
else:
    SHOW_EVERY = 30
    LEARNING_RATE = 1.0
    DISCOUNT = 0.95
    EPSILON = 1.0
    EPSILON_DECREMENT = 0.998
    EPSILON_MIN = 0.01
    EPSILON_MIN2 = 0.4

    max_value_for_Rmax = 100
    conf.ITERATIONS_PER_EPISODE = 30

#--------------------------------------------
'''#parametri marco
SHOW_EVERY = 50
LEARNING_RATE = 0.1 #alpha
DISCOUNT = 0.99 #gamma
EPSILON = 0.1
EPSILON_DECREMENT = 0.99
EPSILON_MIN = 1
EPSILON_MIN2 = 0.01

max_value_for_Rmax = 1
ITERATIONS_PER_EPISODE = 30'''
# _____________________________________________
policy_n = []

#with open("iter.py", "w") as text_file:
#    text_file.write('j =' + '{}'.format(j))

'''os.system('python3 scenario_objects.py')
os.system('python3 plotting.py')
os.system('python3 my_utils.py')
#os.system('python3 load_and_save_data.py')
os.system('python3 restraining_bolt.py')
os.system('python3 temp_wrapper.py')

os.system('python3 ./envs/custom_env_dir/custom_uav_env.py')
os.system('python3 ./temprl/temprl/automata.py')
os.system('python3 ./temprl/temprl/wrapper.py')
os.system('python3 agent.py')'''


#ITERATIONS_PER_EPISODE = iteration_rest()

#Colors, UAVS_POS = UAV_i(UAVS_POS, hosp_pos, Colors, j)

#UAVS_POS = UAVS_POS1[j]
#hosp_pos = [(5, 9), (3, 2), (7,7), (6,6)]
#Colors = [['None', 'gold', 'purple'], ['None', 'orange', 'brown']]
#nb_colors = 2
#UAVS_POS = env.UAVS_POS

'''def changePos_idx(newIdx):
    import my_utils
    my_utils.o = newIdx

changePos_idx(j)'''


# __________________ Main loadings: __________________

env = gym.make('UAVEnv-v0')
MAX_UAV_HEIGHT = env.max_uav_height
n_actions = env.nb_actions
actions_indeces = range(n_actions)
cs_cells = env.cs_cells
cs_cells_coords_for_UAVs = [(cell._x_coord, cell._y_coord) for cell in cs_cells] if conf.DIMENSION_2D==True else [(cell._x_coord, cell._y_coord, cell._z_coord) for cell in cs_cells]
#cells_matrix = env.cells_matrix
action_set_min = env.action_set_min
if (conf.UNLIMITED_BATTERY==False):
    q_table_action_set = env.q_table_action_set
    charging_set = env.charging_set
    come_home_set = env.come_home_set

reset_uavs = env.reset_uavs
reset_priority = env.reset_priority

print(env.max_uav_height, env.nb_actions, range(n_actions), env.cs_cells,
      [(cell._x_coord, cell._y_coord) for cell in cs_cells], action_set_min)
#breakpoint()

plot = plotting.Plot()
centroids = env.cluster_centroids
# Scale centroids according to the selected resolution:
env_centroids = [(centroid[0]/conf.CELL_RESOLUTION_PER_COL, centroid[1]/conf.CELL_RESOLUTION_PER_ROW) for centroid in centroids]


# _____________________________________________________

def show_and_save_info(q_table_init, q_table, dimension_space, battery_type, users_request, reward_func,
                       case_directory):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Save the main settings in 'env_and_train_info.txt' file and show them before training start.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    info = []

    info1 = "\n\n_______________________________________ENVIRONMENT AND TRAINING INFO: _______________________________________\n"
    info.append(info1)

    info2 = "\nTraining:\n"
    info.append(info2)
    info3 = "\nconf.EPISODES: " + str(conf.EPISODES)
    info.append(info3)
    info4 = "\nITERATIONS PER EPISODE: " + str(conf.ITERATIONS_PER_EPISODE)
    info.append(info4)
    info5 = "\nINITIAL EPSILON: " + str(EPSILON)
    info.append(info5)
    info6 = "\nMINIMUM EPSILON: " + str(EPSILON_MIN)
    info.append(info6)
    info31 = "\nEPSILON DECREMENT: " + str(EPSILON_DECREMENT)
    info.append(info31)
    info7 = "\nLEARNING RATE: " + str(LEARNING_RATE)
    info.append(info7)
    info8 = "\nDISCOUNT RATE: " + str(DISCOUNT)
    info.append(info8)
    if (q_table_init == "Max Reward"):
        info9 = "\nQ-TABLE INITIALIZATION: " + q_table_init + " with a Rmax value equal to: " + str(max_value_for_Rmax)
    else:
        info9 = "\nQ-TABLE INITIALIZATION: " + q_table_init
    info.append(info9)
    info28 = "\nQ-TABLE DIMENSION PER UAV: " + str(len(q_table))
    info.append(info28)
    info29 = "\nREWARD FUNCTION USED: " + str(reward_func) + "\n\n"
    info.append(info29)

    info10 = "\nEnvironment:\n"
    info.append(info10)

    if dimension_space == "2D":
        info11 = "\nMAP DIMENSION AT MINIMUM RESOLUTION: " + str(conf.AREA_WIDTH) + "X" + str(conf.AREA_HEIGHT)
        info12 = "\nMAP DIMENSION AT DESIRED RESOLUTION: " + str(conf.CELLS_ROWS) + "X" + str(conf.CELLS_COLS)
        info.append(info11)
        info.append(info12)
    else:
        Z_DIM = MAX_UAV_HEIGHT - conf.MIN_UAV_HEIGHT
        info11 = "\nMAP DIMENSION AT MINIMUM RESOLUTION: " + str(conf.AREA_WIDTH) + "X" + str(
            conf.AREA_HEIGHT) + "X" + str(Z_DIM)
        info12 = "\nMAP DIMENSION AT DESIRED RESOLUTION: " + str(conf.CELLS_ROWS) + "X" + str(
            conf.CELLS_COLS) + "X" + str(Z_DIM)
        info13 = "\nMINIMUM UAVs FLIGHT HEIGHT: " + str(conf.MIN_UAV_HEIGHT)
        info14 = "\nMAXIMUM UAVs FLIGHT HEIGHT: " + str(MAX_UAV_HEIGHT)
        info32 = "\nMINIMUM COVERAGE PERCENTAGE OF OBSTACLES: " + str(conf.MIN_OBS_PER_AREA * 100) + " %"
        info33 = "\nMAXIMUM COVERAGE PERCENTAGE OF OBSTACELS: " + str(conf.MAX_OBS_PER_AREA * 100) + " %"
        info34 = "\nMAXIMUM FLIGHT HEIGHT OF A UAV: " + str(
            MAX_UAV_HEIGHT) + ", equal to the height of the highest obstacle"
        info35 = "\nMINIMUM FLIGHT HEIGHT OF A UAV: " + str(
            conf.MIN_UAV_HEIGHT) + ", equal to the height of the Charging Stations"
        info36 = "\nUAV MOTION STEP ALONG Z-AXIS: " + str(conf.UAV_Z_STEP)
        info.append(info36)
        info.append(info11)
        info.append(info12)
        info.append(info13)
        info.append(info14)
        info.append(info32)
        info.append(info33)
        info.append(info34)
        info.append(info35)
        info.append(info36)
    info15 = "\nUAVs NUMBER: " + str(conf.N_UAVS)
    info.append(info15)
    if (dimension_space == "2D"):
        uavs_coords = [
            "UAV " + str(uav_idx + 1) + ": " + str((env.agents[uav_idx]._x_coord, env.agents[uav_idx]._y_coord)) for
            uav_idx in range(conf.N_UAVS)]
        info16 = "\nUAVs INITIAL COORDINATES: " + str(uavs_coords)
    else:
        uavs_coords = ["UAV " + str(uav_idx + 1) + ": " + str(
            (env.agents[uav_idx]._x_coord, env.agents[uav_idx]._y_coord, env.agents[uav_idx]._z_coord)) for uav_idx in
                       range(conf.N_UAVS)]
        info16 = "\nUAVs INITIAL COORDINATES: " + str(uavs_coords)
    info40 = "\n UAVs FOOTPRINT DIMENSION: " + str(conf.ACTUAL_UAV_FOOTPRINT)
    info.append(info40)
    info.append(info16)
    info17 = "\nUSERS CLUSTERS NUMBER: " + str(len(env.cluster_centroids))
    info30 = "\nUSERS INITIAL NUMBER: " + str(env.n_users)
    info.append(info17)
    info.append(info30)
    centroids_coords = ["CENTROIDS: " + str(centroid_idx + 1) + ": " + str(
        (env.cluster_centroids[centroid_idx][0], env.cluster_centroids[centroid_idx][1])) for centroid_idx in
                        range(len(env.cluster_centroids))]
    info18 = "\nUSERS CLUSTERS PLANE-COORDINATES: " + str(centroids_coords)
    info37 = "\nCLUSTERS RADIUSES: " + str(env.clusters_radiuses)
    info.append(info37)
    info.append(info18)
    info19 = "\nDIMENION SPACE: " + str(dimension_space)
    info.append(info19)
    info20 = "\nBATTERY: " + str(battery_type)
    info.append(info20)
    info21 = "\nUSERS SERVICE TIME REQUEST: " + str(users_request)
    info.append(info21)
    if (conf.STATIC_REQUEST == True):
        info22 = "\nUSERS REQUEST: Static"
    else:
        info22 = "\nUSERS REQUEST: Dynamic"
    info.append(info22)
    if (conf.USERS_PRIORITY == False):
        info23 = "\nUSERS ACCOUNTS: all the same"
    else:
        info23 = "\nUSERS ACCOUNTS: " + str(USERS_ACCOUNTS)
    info.append(info23)
    if (conf.INF_REQUEST == True):
        # If the users service request is infite, then we assume that the UAVs are providing only one service.
        info24 = "\nNUMBER SERVICES PROVIDED BY UAVs: 1"
    else:
        info24 = "\nNUMBER SERVICES PROVIDED BY UAVs: 3"
    info.append(info24)
    if (conf.UNLIMITED_BATTERY == True):
        info25 = "\nCHARGING STATIONS NUMBER: N.D."
    else:
        info25 = "\nCHARGING STATIONS NUMBER: " + str(conf.N_CS)
        info_37 = "\nCHARGING STATIONS COORDINATES: " + str(
            [(cell._x_coord, cell._y_coord, cell._z_coord) for cell in env.cs_cells])
        info.append(info_37)
        info38 = "\nTHRESHOLD BATTERY LEVEL PERCENTAGE CONSIDERED CRITICAL: " + str(conf.PERC_CRITICAL_BATTERY_LEVEL)
        info.append(info38)
        info39 = "\nBATTERY LEVELS WHEN CHARGING SHOWED EVERY " + str(
            conf.SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT) + " CHARGES"
        info.append(info39)
    info.append(info25)
    if (conf.CREATE_ENODEB == True):
        info26 = "\nENODEB: Yes"
    else:
        info26 = "\nENODEB: No"
    info.append(info26)
    info27 = "\n__________________________________________________________________________________________________________________\n\n"
    info.append(info27)

    file = open(join(saving_directory, "env_and_train_info.txt"), "w")

    for i in info:
        print(i)
        file.write(i)

    file.close()

    # time.sleep(5)


def compute_subareas(area_width, area_height, x_split, y_split):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Return 'xy_limits' and middle points for some subareas (oo the all considered map) which can be used for a Q-table initialization based on 'a priori knowledge'.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    subW_min = subH_min = 0
    subW_max = area_width / x_split
    subH_max = area_height / y_split
    subareas_xy_limits = []
    subareas_middle_points = []

    for x_subarea in range(1, x_split + 1):
        W_max = subW_max * x_subarea

        for y_subarea in range(1, y_split + 1):
            H_max = subH_max * y_subarea

            x_limits = (subW_min, W_max)
            y_limits = (subH_min, H_max)
            subareas_xy_limits.append([x_limits, y_limits])
            # Compute the middle point of each subarea:
            subareas_middle_points.append((subW_min + (W_max - subW_min) / 2, subH_min + (H_max - subH_min) / 2))

            subH_min = H_max

        subW_min = W_max
        subH_min = 0

    return subareas_xy_limits, subareas_middle_points


def compute_prior_rewards(agent_pos_xy, best_prior_knowledge_points):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Compute the values used to inizialized the Q-table based on 'a priori initialization'.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    actions = env.q_table_action_set
    # Initialize a random agent just to use easily the 'move' methods for 2D and 3D cases:
    agent_test = agent.Agent((agent_pos_xy[0], agent_pos_xy[1], 0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # Compute all the possible positions according to the available actions
    if (conf.DIMENSION_2D == True):
        # Prior knowledge obvoiusly does not take into account the battery level:
        new_agent_pos_per_action = [agent_test.move_2D_unlimited_battery((agent_pos_xy[0], agent_pos_xy[1]), action) for
                                    action in actions]
    else:
        new_agent_pos_per_action = [
            agent_test.move_3D_unlimited_battery((agent_pos_xy[0], agent_pos_xy[1], agent_pos_xy[2]), action) for action
            in actions]
    prior_rewards = []
    for pos in new_agent_pos_per_action:
        current_distances_from_best_points = [LA.norm(np.array([pos[0], pos[1]]) - np.array(best_point)) for best_point
                                              in best_prior_knowledge_points]
        # The reference distance for the reward in the current state is based on the distance between the agent and the closer 'best point':
        current_reference_distance = min(current_distances_from_best_points)
        current_normalized_ref_dist = current_reference_distance / diagonal_area_value
        prior_rewards.append(1 - current_normalized_ref_dist)

    return prior_rewards


# _________________________________ Check if Q-table initialization is based on 'a prior knoledge' _________________________________

if (conf.PRIOR_KNOWLEDGE == True):
    subareas_limits, subareas_middle_points = compute_subareas(conf.CELLS_COLS, conf.CELLS_ROWS, conf.X_SPLIT,
                                                               conf.Y_SPLIT)
    best_prior_knowledge_points = []
    diagonal_area_value = sqrt(pow(conf.CELLS_ROWS, 2) + pow(conf.CELLS_COLS,
                                                             2))  # --> The diagonal is the maximum possibile distance between two points, and then it will be used to normalize the distance when the table is initialized under the assumption of 'prior knowledge'. 0.5 is used because the UAV is assumed to move from the middle of a cell to the middle of another one.

    for centroid in env_centroids:
        centroid_x = centroid[0]
        centroid_y = centroid[1]

        for subarea in range(conf.N_SUBAREAS):
            current_subarea = subareas_limits[subarea]

            if (((centroid_x >= current_subarea[0][0]) and (centroid_x < current_subarea[0][1])) and (
                    (centroid_y >= current_subarea[1][0]) and (centroid_y < current_subarea[1][1]))):
                best_prior_knowledge_points.append(subareas_middle_points[subarea])

DEFAULT_CLOSEST_CS = (None, None, None) if conf.DIMENSION_2D == False else (None, None)


# ___________________________________________________________________________________________________________________________________

def go_to_recharge(action, agent):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Perform SIDE EFFECT on '_path_to_the_closest_CS' and '_current_pos_in_path_to_CS'.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # agent._coming_home is set equal to True inside 'move' method, which is called inside 'step' method (just after 'choose_action'). Thus, here it is not needed.
    closest_CS = agent._cs_goal
    # Compute the path to the closest CS only if needed:
    if (closest_CS == DEFAULT_CLOSEST_CS):
        _ = agent.compute_distances(
            cs_cells)  # --> Update the closest CS just in case the agent need to go the CS (which is obviously the closest one).
        agent._path_to_the_closest_CS = astar(env.cells_matrix, env.get_agent_pos(agent), agent._cs_goal)
        agent._current_pos_in_path_to_CS = 0  # --> when compute the path, set to 0 the counter indicating the current position (which belongs to the computed path) you have to go.


def choose_action(uavs_q_tables, which_uav, obs, agent, battery_in_CS_history, cells_matrix):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Return the chosen action according to the current set of action which depends on the considered scenario, battery level and other parameters:   #
    # SIDE EFFECT on different agents attributes are performed.                                                                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if ((ANALYZED_CASE == 1) or (ANALYZED_CASE == 3)):  # --> UNLIMITED BATTERY:
        # print("obsssssssssssssssssssssssssssssssssssss", obs)
        if (conf.HOSP_SCENARIO == False):
            obs = tuple([round(ob, 1) for ob in obs])

    else:  # --> LIMITED BATTERY:
        if (conf.HOSP_SCENARIO == False):
            coords = tuple([round(ob, 1) for ob in obs[0]])
            obs = tuple([coords, obs[1]])

    # obs = tuple([round(Decimal(ob), 1) for ob in obs])
    # print("weeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", obs)
    all_actions_values = [values for values in uavs_q_tables[which_uav][obs]]
    current_actions_set = agent._action_set

    if (conf.UAV_STANDARD_BEHAVIOUR == False):

        if conf.UNLIMITED_BATTERY == False:
            if (current_actions_set == action_set_min):  # == ACTION_SPACE_3D_MIN
                all_actions_values[GO_TO_CS_INDEX] = -inf
                all_actions_values[CHARGE_INDEX] = -inf

            elif (current_actions_set == come_home_set):  # == ACTION_SPACE_3D_COME_HOME

                if ((agent._coming_home == True) and (env.get_agent_pos(agent) != agent._cs_goal)):
                    action = GO_TO_CS_INDEX

                    if (conf.Q_LEARNING == True):
                        agent._current_pos_in_path_to_CS += 1

                    return action

                elif (agent._coming_home == False):
                    all_actions_values[CHARGE_INDEX] = -inf
                    agent._required_battery_to_CS = agent.needed_battery_to_come_home()

                elif ((agent._coming_home == True) and agent.check_if_on_CS()):
                    agent._n_recharges += 1
                    n_recharges = agent._n_recharges

                    if (n_recharges % conf.SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT == 0):
                        battery_in_CS_history.append(agent._battery_level)

                    action = CHARGE_INDEX

                    return action

            elif (current_actions_set == charging_set):  # == ACTION_SPACE_3D_WHILE_CHARGING

                if ((agent.check_if_on_CS()) and (agent._battery_level < conf.FULL_BATTERY_LEVEL)):
                    action = CHARGE_INDEX
                    # agent._coming_home is set equal to True inside 'move' method, which is called inside 'step' method (just after 'choose_action'). Thus, here it is not needed.

                    return action

                elif (agent._battery_level >= conf.FULL_BATTERY_LEVEL):
                    agent._battery_level = conf.FULL_BATTERY_LEVEL
                    all_actions_values[CHARGE_INDEX] = -inf
                    all_actions_values[GO_TO_CS_INDEX] = -inf

    rand = np.random.random()

    if (conf.UAV_STANDARD_BEHAVIOUR == False):
        if (rand > EPSILON):
            # Select the best action so far (based on the current action set of the agent, assign actions values to a reduce action set):
            action = np.argmax(all_actions_values)
            # action = np.argmax(uavs_q_tables[0][obs])
            # print(action, "action")
        else:
            n_actions_to_not_consider = n_actions - agent._n_actions
            n_current_actions_to_consider = n_actions - n_actions_to_not_consider
            prob_per_action = 1 / n_current_actions_to_consider
            probabilities = [prob_per_action if act_idx < n_current_actions_to_consider else 0.0 for act_idx in
                             actions_indeces]
            # Select the action randomly:
            action = np.random.choice(actions_indeces, p=probabilities)
            # action = np.random.randint(0, 4)
    else:

        if (conf.UNLIMITED_BATTERY == False):

            if ((agent._coming_home == True) and (env.get_agent_pos(agent) in cs_cells_coords_for_UAVs)):
                action = conf.CHARGE

            elif ((agent._charging == True) and (agent._battery_level < conf.FULL_BATTERY_LEVEL)):
                action = conf.CHARGE

            elif (agent._battery_level <= conf.CRITICAL_BATTERY_LEVEL):
                action = GO_TO_CS
                agent._current_pos_in_path_to_CS += 1
                go_to_recharge(GO_TO_CS_INDEX, agent)

            elif (which_uav == 2):
                action = agent.action_for_standard_h(env.cells_matrix)

            elif (which_uav == 1):
                action = agent.action_for_standard_v(env.cells_matrix)

            elif (which_uav == 0):
                action = agent.action_for_standard_square_clockwise(env.cells_matrix)

        else:

            if (which_uav == 2):
                action = agent.action_for_standard_h(env.cells_matrix)

            elif (which_uav == 1):
                action = agent.action_for_standard_v(env.cells_matrix)

            elif (which_uav == 0):
                action = agent.action_for_standard_square_clockwise(env.cells_matrix)

        return action

    if (action == GO_TO_CS_INDEX):
        go_to_recharge(action, agent)

    return action


# ________________________________________ Assign a number_id to each analyzed case in order to initialize the proper Q-table based on te current case: ________________________________________

ANALYZED_CASE = 0

if ((conf.USERS_PRIORITY == False) and (conf.CREATE_ENODEB == False)):
    # 2D case with UNLIMITED UAVs battery autonomy:
    if ((conf.DIMENSION_2D == True) and (conf.UNLIMITED_BATTERY == True)):
        ANALYZED_CASE = 1
        considered_case_directory = "2D_un_bat"
        dimension_space = "2D"
        battery_type = "Unlimited"
        reward_func = "Reward function 1"

    # 2D case with LIMITED UAVs battery autonomy:
    elif ((conf.DIMENSION_2D == True) and (conf.UNLIMITED_BATTERY == False)):
        ANALYZED_CASE = 2
        considered_case_directory = "2D_lim_bat"
        dimension_space = "2D"
        battery_type = "Limited"
        reward_func = "Reward function 2"

    # 3D case with UNLIMITED UAVs battery autonomy:
    elif ((conf.DIMENSION_2D == False) and (conf.UNLIMITED_BATTERY == True)):
        ANALYZED_CASE = 3
        considered_case_directory = "3D_un_bat"
        dimension_space = "3D"
        battery_type = "Unlimited"
        reward_func = "Reward function 1"

    # 3D case with LIMITED UAVs battery autonomy:
    elif ((conf.DIMENSION_2D == False) and (conf.UNLIMITED_BATTERY == False)):
        ANALYZED_CASE = 4
        considered_case_directory = "3D_lim_bat"
        dimension_space = "3D"
        battery_type = "Limited"
        reward_func = "Reward function 2"

    if (conf.INF_REQUEST == True):
        # setting_not_served_users = agent.Agent.set_not_served_users_inf_request
        service_request_per_epoch = env.n_users * conf.ITERATIONS_PER_EPISODE
        considered_case_directory += "_inf_req"
        users_request = "Continue"
    else:
        # setting_not_served_users = agent.Agent.set_not_served_users
        service_request_per_epoch = 0  # --> TO SET --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        considered_case_directory += "_lim_req"
        users_request = "Discrete"
        if (conf.MULTI_SERVICE == True):
            considered_case_directory += "_multi_service_limited_bandwidth"
            reward_func = "Reward function 3"

else:

    # conf.CREATE_ENODEB = False
    # conf.DIMENSION_2D = False
    # conf.UNLIMITED_BATTERY = True
    # conf.INF_REQUEST = True
    # conf.STATIC_REQUEST = True
    # conf.USERS_PRIORITY = False
    assert False, "Environment parameters combination not implemented yet: conf.STATIC_REQUEST: %s, conf.DIMENSION_2D: %s, conf.UNLIMITED_BATTERY: %s, conf.INF_REQUEST: %s, conf.USERS_PRIORITY: %s, conf.CREATE_ENODEB: %s" % (
    conf.STATIC_REQUEST, conf.DIMENSION_2D, conf.UNLIMITED_BATTERY, conf.INF_REQUEST, conf.USERS_PRIORITY,
    conf.CREATE_ENODEB)
    pass  # TO DO --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ______________________________________________________________________________________________________________________________________________________________________________________________


# ________________________________________ Directory cration to save images and data: ________________________________________

considered_case_directory += "_" + str(conf.N_UAVS) + "UAVs" + "_" + str(len(env.cluster_centroids)) + "clusters"

cases_directory = "Cases"
if (conf.R_MAX == True):
    sub_case_dir = "Max Initialization"
    q_table_init = "Max Reward"
elif (conf.PRIOR_KNOWLEDGE == True):
    sub_case_dir = "Prior Initialization"
    q_table_init = "Prior Knowledge"
else:
    sub_case_dir = "Random Initialization"
    q_table_init = "Random Reward"

saving_directory = join(cases_directory, considered_case_directory, sub_case_dir)

if not isdir(cases_directory): mkdir(cases_directory)
if not isdir(join(cases_directory, considered_case_directory)): mkdir(join(cases_directory, considered_case_directory))
if not isdir(saving_directory): mkdir(saving_directory)

# _____________________________________________________________________________________________________________________________


# ________________________________________________________________________________ Q-table initialization: ________________________________________________________________________________________________

map_width = conf.CELLS_COLS
map_length = conf.CELLS_ROWS
map_height = conf.MAXIMUM_AREA_HEIGHT

# Agents initialization:
agents = env.agents

# It is called Q-table, but if conf.SARSA algorithm is used, it will be actually a conf.SARSA-table:
uavs_q_tables = None
uavs_e_tables = None
if uavs_q_tables is None:

    print("Q-TABLES INITIALIZATION . . .")
    uavs_q_tables = [None for uav in range(conf.N_UAVS)]
    explored_states_q_tables = [None for uav in range(conf.N_UAVS)]
    uavs_e_tables = [None for uav in range(conf.N_UAVS)]
    uav_counter = 0
    # print(uavs_q_tables)
    # breakpoint()
    for uav in range(conf.N_UAVS):
        current_uav_q_table = {}
        current_uav_explored_table = {}

        for x_agent in np.arange(0, map_width + 1, 1):  # range(map_width)

            for y_agent in np.arange(0, map_length + 1, 1):  # range(map_length)

                x_agent = round(x_agent, 1)
                y_agent = round(y_agent, 1)

                # 2D case with UNLIMITED UAVs battery autonomy:
                if (ANALYZED_CASE == 1):

                    if (conf.PRIOR_KNOWLEDGE == True):
                        prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                        current_uav_q_table[(x_agent, y_agent)] = [prior_rewards[action] for action in range(n_actions)]
                    elif (conf.R_MAX == True):
                        current_uav_q_table[(x_agent, y_agent)] = [max_value_for_Rmax for action in range(n_actions)]
                    elif (conf.HOSP_SCENARIO == True):
                        for prior in conf.HOSP_PRIORITIES:
                            # current_uav_q_table[((x_agent, y_agent), 1, prior)] = [np.random.uniform(0, 1) for action in range(n_actions)]   #da vedere se mettere 0
                            # current_uav_q_table[((x_agent, y_agent), 0, prior)] = [np.random.uniform(0, 1) for action in range(n_actions)]  # da vedere se mettere 0
                            current_uav_q_table[((x_agent, y_agent), 1, prior)] = [0 for action in range(
                                n_actions)]  # da vedere se mettere 0
                            current_uav_q_table[((x_agent, y_agent), 0, prior)] = [0 for action in range(
                                n_actions)]  # da vedere se mettere 0
                        # current_uav_q_table[((x_agent, y_agent), 1, 0)] = [np.random.uniform(0, 1) for action in range(n_actions)]
                        # current_uav_q_table[((x_agent, y_agent), 0, 0)] = [np.random.uniform(0, 1) for action in range(n_actions)]
                        current_uav_q_table[((x_agent, y_agent), 1, 0)] = [0 for action in range(n_actions)]
                        current_uav_q_table[((x_agent, y_agent), 0, 0)] = [0 for action in range(n_actions)]
                    else:
                        current_uav_q_table[(x_agent, y_agent)] = [np.random.uniform(0, 1) for action in
                                                                   range(n_actions)]

                    current_uav_explored_table[(x_agent, y_agent)] = [False for action in range(n_actions)]

                # 2D case with LIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 2):
                    for battery_level in np.arange(0, conf.FULL_BATTERY_LEVEL + 1, conf.PERC_CONSUMPTION_PER_ITERATION):

                        if (conf.PRIOR_KNOWLEDGE == True):
                            prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                            current_uav_q_table[((x_agent, y_agent), battery_level)] = [(1 - prior_rewards) for action
                                                                                        in range(n_actions)]
                        elif (conf.R_MAX == True):
                            current_uav_q_table[((x_agent, y_agent), battery_level)] = [max_value_for_Rmax for action in
                                                                                        range(n_actions)]
                        else:
                            current_uav_q_table[((x_agent, y_agent), battery_level)] = [np.random.uniform(0, 1) for
                                                                                        action in range(n_actions)]

                        current_uav_explored_table[(x_agent, y_agent), battery_level] = [False for action in
                                                                                         range(n_actions)]

                # 3D case with UNLIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 3):
                    for z_agent in range(conf.MIN_UAV_HEIGHT, MAX_UAV_HEIGHT, conf.UAV_Z_STEP):

                        if (conf.PRIOR_KNOWLEDGE == True):
                            prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [(1 - prior_rewards) for action in
                                                                                range(n_actions)]
                        elif (conf.R_MAX == True):
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [max_value_for_Rmax for action in
                                                                                range(n_actions)]
                        else:
                            current_uav_q_table[(x_agent, y_agent, z_agent)] = [np.random.uniform(0, 1) for action in
                                                                                range(n_actions)]

                        current_uav_explored_table[(x_agent, y_agent, z_agent)] = [False for action in range(n_actions)]

                # 3D case with LIMITED UAVs battery autonomy:
                elif (ANALYZED_CASE == 4):
                    if (conf.UAV_STANDARD_BEHAVIOUR == False):
                        range_for_z = range(conf.MIN_UAV_HEIGHT, MAX_UAV_HEIGHT, conf.UAV_Z_STEP)
                    else:
                        range_for_z = range(conf.MIN_UAV_HEIGHT, MAX_UAV_HEIGHT + conf.UAV_Z_STEP + 1, conf.UAV_Z_STEP)
                    for z_agent in range_for_z:

                        for battery_level in np.arange(0, conf.FULL_BATTERY_LEVEL + 1,
                                                       conf.PERC_CONSUMPTION_PER_ITERATION):

                            if (conf.PRIOR_KNOWLEDGE == True):
                                prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [(1 - prior_rewards)
                                                                                                     for action in
                                                                                                     range(n_actions)]
                            elif (conf.R_MAX == True):
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [max_value_for_Rmax
                                                                                                     for action in
                                                                                                     range(n_actions)]
                            else:
                                current_uav_q_table[((x_agent, y_agent, z_agent), battery_level)] = [
                                    np.random.uniform(0, 1) for action in range(n_actions)]

                            current_uav_explored_table[(x_agent, y_agent, z_agent), battery_level] = [False for action
                                                                                                      in
                                                                                                      range(n_actions)]

        uavs_q_tables[uav] = current_uav_q_table
        # uavs_e_tables[uav] = current_uav_q_table
        if conf.HOSP_SCENARIO == False:
            explored_states_q_tables[uav] = current_uav_explored_table
        print("Q-Table for Uav ", uav, " created")

    print("Q-TABLES INITIALIZATION COMPLETED.")
else:

    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# _________________________________________________________________________________________________________________________________________________________________________________________________________

show_and_save_info(q_table_init, uavs_q_tables[0], dimension_space, battery_type, users_request, reward_func,
                   saving_directory)

q_tables_dir = "QTables"
q_tables_directory = join(saving_directory, q_tables_dir)

uav_ID = "UAV"
uavs_directories = [0 for uav in range(conf.N_UAVS)]
q_tables_directories = [0 for uav in range(conf.N_UAVS)]
for uav in range(1, conf.N_UAVS + 1):
    current_uav_dir = join(saving_directory, uav_ID + str(uav))
    if not isdir(current_uav_dir): mkdir(current_uav_dir)
    uavs_directories[uav - 1] = current_uav_dir
    current_q_table_dir = join(current_uav_dir, q_tables_dir)
    q_tables_directories[uav - 1] = join(current_uav_dir, q_tables_dir)
    if not isdir(current_q_table_dir): mkdir(current_q_table_dir)
uav_directory = uav_ID

# ________________________________________ Variable initialization before training loop: __________________________________________
best_reward_episode = [[] for uav in range(conf.N_UAVS)]
uavs_episode_rewards = [[] for uav in range(conf.N_UAVS)]
users_in_foots = [[] for uav in range(conf.N_UAVS)]
best_policy = [0 for uav in range(conf.N_UAVS)]
best_policy_obs = [[] for uav in range(conf.N_UAVS)]
obs_recorder = [[() for it in range(conf.ITERATIONS_PER_EPISODE)] for uav in range(conf.N_UAVS)]

avg_QoE1_per_epoch = [0 for ep in range(conf.EPISODES)]
avg_QoE2_per_epoch = [0 for ep in range(conf.EPISODES)]
avg_QoE3_per_epoch = [0 for ep in range(conf.EPISODES)]

q_values = [[] for episode in range(conf.N_UAVS)]

if (conf.DIMENSION_2D == True):
    GO_TO_CS_INDEX = conf.GO_TO_CS_2D_INDEX
    CHARGE_INDEX = conf.CHARGE_2D_INDEX
else:
    CHARGE_INDEX = conf.CHARGE_3D_INDEX
    CHARGE_INDEX_WHILE_CHARGING = conf.CHARGE_3D_INDEX_WHILE_CHARGING
    GO_TO_CS_INDEX = conf.GO_TO_CS_3D_INDEX
    GO_TO_CS_INDEX_HOME_SPACE = conf.GO_TO_CS_3D_INDEX_HOME_SPACE

# time.sleep(5)

epsilon_history = [0 for ep in range(conf.EPISODES)]
crashes_history = [0 for ep in range(conf.EPISODES)]
battery_in_CS_history = [[] for uav in range(conf.N_UAVS)]
n_active_users_per_epoch = [0 for ep in range(conf.EPISODES)]
provided_services_per_epoch = [[0, 0, 0] for ep in range(conf.EPISODES)]
n_active_users_per_episode = [0 for ep in range(conf.EPISODES)]
UAVs_used_bandwidth = [[0 for ep in range(conf.EPISODES)] for uav in range(conf.N_UAVS)]
users_bandwidth_request_per_UAVfootprint = [[0 for ep in range(conf.EPISODES)] for uav in range(conf.N_UAVS)]

MOVE_USERS = False

# TEMPORAL GOAL --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if (conf.HOSP_SCENARIO == True):
    print("Computing the temporal goal . . .")
    tg = [RestrainingBolt.make_uavs_goal() for uav in range(conf.N_UAVS)]  # tg = RestrainingBolt.make_uavs_goal()
    # print("GUARDA QUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA :", tgs)
    env = UAVsTemporalWrapper(env, temp_goals=tg)  # env = UAVsTemporalWrapper(env, temp_goals=[tg])
    print("Temporal goal computed.")

# q_values_current_episode = [0 for uav in range(conf.N_UAVS)]
# ________________________________________________________________________________ Training start: _____________________________________________________________________________________________________________

print("\nSTART TRAINING . . .\n")
for episode in range(1, conf.EPISODES + 1):
    # Guarda qui
    # reset_uavs(agents[0])  #Da decidere se fare due funzioni separate per batteria e reset della posizione ad ogni episodio
    # reset_uavs(agents[1])
    if conf.HOSP_SCENARIO == True:
        reset_priority()
        env.reset()

    else:
        if (conf.STATIC_REQUEST == False):
            if (episode % conf.MOVE_USERS_EACH_N_EPOCHS == 0):
                env.compute_users_walk_steps()
                MOVE_USERS = True
            else:
                MOVE_USERS = False

        if (conf.INF_REQUEST == False):
            if (episode % conf.UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS == 0):
                env.update_users_requests(env.users)

    epsilon_history[episode - 1] = EPSILON

    print("| EPISODE: {ep:3d} | Epsilon: {eps:6f}".format(ep=episode, eps=EPSILON), "BEST POLICY:", best_policy)

    QoEs_store = [[], []]  # --> Store QoE1 and QoE2
    current_QoE3 = 0
    users_served_time = 0
    users_request_service_elapsed_time = 0
    current_provided_services = [0, 0, 0]
    current_UAV_bandwidth = [0 for uav in range(conf.N_UAVS)]
    current_requested_bandwidth = [0 for uav in range(conf.N_UAVS)]
    uavs_episode_reward = [0 for uav in range(conf.N_UAVS)]
    crashes_current_episode = [False for uav in range(conf.N_UAVS)]
    q_values_current_episode = [0 for uav in range(conf.N_UAVS)]
    n_active_users_current_it = 0
    tr_active_users_current_it = 0
    ec_active_users_current_it = 0
    dg_active_users_current_it = 0
    if conf.SARSA_lambda == True:
        for i in uavs_e_tables[0]:
            # print(i, uavs_e_tables[0][i])
            uavs_e_tables[0][i] = [0 for action in range(5)]
        # Inizializzo la e_tables a ogni episodio
    # 30minutes
    for i in range(conf.ITERATIONS_PER_EPISODE):
        # print("\nITERATIONS: ", i, "----------------------------------------------")

        if conf.HOSP_SCENARIO == False:
            if (conf.INF_REQUEST == True):
                n_active_users = env.n_users

            else:
                n_active_users, tr_active_users, ec_active_users, dg_active_users, n_tr_served, n_ec_served, n_dg_served = env.get_active_users()
                tr_active_users_current_it += tr_active_users
                ec_active_users_current_it += ec_active_users
                dg_active_users_current_it += dg_active_users

            n_active_users_current_it += n_active_users

            if (MOVE_USERS == True):
                env.move_users(i + 1)

            env.all_users_in_all_foots = []

        for UAV in range(conf.N_UAVS):
            # print("ID AGENTE: ", agents[UAV]._uav_ID)
            # Skip analyzing the current UAV features until it starts to work (you can set a delayed start for each uav):
            current_iteration = i + 1

            if (episode == 1):
                # Case in which are considered the UAVs from the 2-th to the n-th:
                if (UAV > 0):
                    if (current_iteration != (conf.DELAYED_START_PER_UAV * (UAV))):
                        pass
                        # continue

                # drone_pos = (UAVS_POS[0][0], UAVS_POS[0][1])
            # env.agents_paths[UAV][i] = env.get_agent_pos(agents[UAV])
            if (conf.NOISE_ON_POS_MEASURE == True):
                drone_pos = env.noisy_measure_or_not(env.get_agent_pos(agents[UAV]))
            else:
                drone_pos = env.get_agent_pos(agents[UAV])  # env.agents_paths[UAV][i]

            # print(drone_pos, "drone_posdrone_posdrone_posdrone_posdrone_posdrone_posdrone_posdrone_posdrone_posdrone_posdrone_pos")
            if (conf.HOSP_SCENARIO == True):
                if (conf.UNLIMITED_BATTERY == True):
                    obs = (
                    drone_pos, agents[UAV].beep, hosp_f.get_color_id(env.cells_matrix[drone_pos[1]][drone_pos[0]]._priority))
                else:
                    obs = (drone_pos, agents[UAV]._battery_level, agents[UAV].beep,
                           hosp_f.get_color_id(env.cells_matrix[drone_pos[1]][drone_pos[0]]._priority))
            else:
                if (conf.UNLIMITED_BATTERY == True):
                    obs = (drone_pos)
                else:
                    obs = (drone_pos, agents[
                        UAV]._battery_level)  # --> The observation will be different when switch from 2D to 3D scenario and viceversa.

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Questa 'obs' viene passata a 'choose_action' senza 'beep' e senza colore prioritario (perhè in 'choose_action' questi due valori non vengono ancora considerati);
            # 'beep' e il colore prioritario vengono invece restituiti in 'obs_' da 'step_agent(..)'.
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            if (conf.UNLIMITED_BATTERY == False):
                env.set_action_set(agents[UAV])

            action = choose_action(uavs_q_tables, UAV, obs, agents[UAV], battery_in_CS_history[UAV], env.cells_matrix)

            obs_, reward, done, info = env.step_agent(agents[UAV], action)
            ##print(reward, "env_wrapper")

            # -----------------------------------------------NO-INTERFERENCE-REWARD--------------------------------------------------
            if conf.NO_INTERFERENCE_REWARD == True:
                drone_pos_ = list(drone_pos)

                if drone_pos_ in traj_csv:
                    reward += -1.0
                    # print("drone_pos", drone_pos, "\n traj_csv", traj_csv)
            # -----------------------------------------------------------------------------------------------------------------------

            if conf.HOSP_SCENARIO == False:
                crashes_current_episode[UAV] = agents[UAV]._crashed

            print(" - Iteration: {it:1d} - Reward per UAV {uav:1d}: {uav_rew:6f}".format(it=i + 1, uav=UAV + 1,
                                                                                         uav_rew=reward), end="\r",
                  flush=True)

            if (conf.UAV_STANDARD_BEHAVIOUR == True):
                action = conf.ACTION_SPACE_STANDARD_BEHAVIOUR.index(action)

            if ((ANALYZED_CASE == 1) or (ANALYZED_CASE == 3)):  # --> UNLIMITED BATTERY

                if (conf.HOSP_SCENARIO == False):
                    obs = tuple([round(ob, 1) for ob in obs])
                    obs_ = tuple([round(ob, 1) for ob in obs_])
                    # obs = tuple([round(ob, 1) for ob in obs])
                    # obs_ = tuple([round(ob, 1) for ob in obs_[0][0]]) #CASO MARCO
                    # obs_ = tuple([round(ob, 1) for ob in (obs_[0][0]['x'], obs_[0][0]['y'])]) #CASO MARCO
                else:
                    # print("OBSSSSSSSSSSSSSSSSSSSSSSSSSS", obs)

                    obs_ = ((obs_[0][0]['x'], obs_[0][0]['y']), obs_[0][0]['beep'], obs_[0][0]['color'])

            else:  # --> LIMITED BATTERY
                coords = tuple([round(ob, 1) for ob in obs[0]])
                obs = tuple([coords, obs[1]])
                coords_ = tuple([round(ob, 1) for ob in obs_[0]])
                obs_ = tuple([coords, obs_[1]])

            if conf.HOSP_SCENARIO == False:
                if not explored_states_q_tables[UAV][obs_][action]:
                    explored_states_q_tables[UAV][obs_][action] = True
            # print("1111111111111111111111111111111111", obs, obs_)
            if conf.HOSP_SCENARIO == True:
                obs_recorder[UAV][i] = obs_  # conf.HOSP_SCENARIO == FALSE: obs_recorder[UAV][i] = obs
                # print("1 - obs_recorder[UAV]", obs_recorder[UAV])
                uavs_episode_reward[UAV] += reward
            else:
                obs_recorder[UAV][i] = obs

            if (conf.UNLIMITED_BATTERY == False):
                if (info == "IS CHARGING"):
                    # print("sbagliatoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
                    # breakpoint()
                    if uavs_episode_reward[UAV] > best_policy[UAV]:
                        best_policy[UAV] = uavs_episode_reward[UAV]
                        best_policy_obs[UAV] = obs_recorder[UAV]

                    obs_recorder[UAV] = [() for i in range(conf.ITERATIONS_PER_EPISODE)]
                    continue

                else:  # --> i.e., crashed case:
                    obs_recorder[UAV] = [() for i in range(conf.ITERATIONS_PER_EPISODE)]
            else:
                # print("0 - best_policy[UAV]", best_policy[UAV], ">", "uavs_episode_reward[UAV]???", uavs_episode_reward[UAV])
                if conf.HOSP_SCENARIO == True:
                    if (current_iteration == conf.ITERATIONS_PER_EPISODE or done):

                        # print(uavs_episode_reward[UAV], best_policy[UAV])
                        if uavs_episode_reward[UAV] > best_policy[UAV]:
                            # print("2 - obs_recorder[UAV]", obs_recorder[UAV])
                            best_policy[UAV] = uavs_episode_reward[UAV]
                            best_policy_obs[UAV] = obs_recorder[UAV]
                            # print("2 - best_policy[UAV]", best_policy[UAV])
                            # print("SALVO LA POLICY !!!!!!!!!!!!!!!!!!!!!!!!!")
                        obs_recorder[UAV] = [() for i in range(conf.ITERATIONS_PER_EPISODE)]
                        # print("obs_recorder[UAV]", obs_recorder[UAV])
                        # print("best_policy_obs[UAV]", best_policy_obs[UAV])
                        # print("best_policy[UAV]", best_policy[UAV])
                    # print("best_policy_obs[UAV]", best_policy_obs[UAV])
                else:
                    if (current_iteration == conf.ITERATIONS_PER_EPISODE):

                        if uavs_episode_reward[UAV] > best_policy[UAV]:
                            best_policy[UAV] = uavs_episode_reward[UAV]
                            best_policy_obs[UAV] = obs_recorder[UAV]

                        obs_recorder[UAV] = [() for i in range(conf.ITERATIONS_PER_EPISODE)]

            # Set all the users which could be no more served after the current UAV action (use different arguments according to the considered case!!!!!!!!!!!!!!!!!!):
            if conf.HOSP_SCENARIO == False:
                agent.Agent.set_not_served_users(env.users, env.all_users_in_all_foots, UAV + 1, QoEs_store, i + 1,
                                                 current_provided_services)
                # setting_not_served_users(env.users, env.all_users_in_all_foots, UAV+1, QoEs_store, i+1) # --> This make a SIDE_EFFECT on users by updating their info.

            if not done or conf.HOSP_SCENARIO == False:
                if (conf.Q_LEARNING == True):
                    # print("obs____________________________", obs_)
                    # print("obssssssssssssssssssssssssssssss", obs)
                    max_future_q = np.max(uavs_q_tables[UAV][obs_])
                    if conf.HOSP_SCENARIO == True:
                        current_q = uavs_q_tables[UAV][obs_][action]
                    else:
                        current_q = uavs_q_tables[UAV][obs_][action]
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                else:

                    if (conf.SARSA == True):
                        action_ = choose_action(uavs_q_tables, UAV, obs_, agents[UAV], battery_in_CS_history[UAV],
                                                env.cells_matrix)
                        future_reward = uavs_q_tables[UAV][obs_][action_]
                        current_q = uavs_q_tables[UAV][obs_][action]
                        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_reward)

                    if (conf.SARSA_lambda == True):
                        """Implements conf.SARSA-Lambda action value function update.
                                e(s,a) = lambda*gamma*e(s,a) + 1(s_t = s, a_t = a).
                                delta_t = reward_t + gamma*q(s_t+1, a_t+1) - q(s_t, a_t).
                                q(s,a) = q(s,a) + alpha*e(s,a)*delta_t.
                                Here we assume gamma=1 (undiscounted).
                                alpha = 1 / N(s,a)."""
                        lambda_value = 0.9
                        action_ = choose_action(uavs_q_tables, UAV, obs_, agents[UAV], battery_in_CS_history[UAV],
                                                env.cells_matrix)
                        future_reward = uavs_q_tables[UAV][obs_][action_]

                        current_q = uavs_q_tables[UAV][obs][action]
                        current_e = uavs_e_tables[UAV][obs][action]

                        # -------------------------PROVA1----------------------------------------------------------------------------

                        '''delta = reward + DISCOUNT * future_reward - current_q
                        print("uavs_e_tables[UAV][obs][action]", uavs_e_tables[UAV][obs][action])
                        current_e += 1
                        print("uavs_e_tables[UAV][obs][action]", uavs_e_tables[UAV][obs][action])

                        new_q = current_q + (1 - LEARNING_RATE) * delta * current_e
                        current_e = DISCOUNT * lambda_value * current_e'''

                        # -------------------------PROVA2----------------------------------------------------------------------------
                        print("ooooo", current_q)
                        # Computing the error
                        delta = reward + DISCOUNT * future_reward - current_q
                        # Setting the eligibility traces
                        print("222222", uavs_e_tables[UAV][obs])
                        uavs_e_tables[UAV][obs] = [i * DISCOUNT * lambda_value for i in uavs_e_tables[UAV][obs]]
                        print("333333", uavs_e_tables[UAV][obs])
                        uavs_e_tables[UAV][obs][action] = 1
                        # Updating the Q values
                        q_tabl = [i * (1 - LEARNING_RATE) * delta for i in uavs_e_tables[UAV][obs]]
                        uavs_q_tables[UAV][obs] = [x
                                                   + y for x, y in zip(uavs_q_tables[UAV][obs], q_tabl)]
                        print("44444", uavs_q_tables[UAV][obs])
                        print("55555", uavs_e_tables[UAV][obs])
                        print("fine", uavs_e_tables[UAV][obs][action])
                        # uavs_e_tables[UAV][obs][action] = uavs_e_tables[UAV][obs][action] + (1 - LEARNING_RATE) * delta

                        # breakpoint()

                        '''for obs in (uavs_q_tables[UAV][obs]):

                            for action in (uavs_q_tables[UAV][action]):
                                print(uavs_q_tables[UAV][obs][action])

                                uavs_q_tables[UAV][s][a] += (1 - LEARNING_RATE) * delta * uavs_e_tables[UAV][s][a]
                                uavs_e_tables[UAV][s][a] = DISCOUNT * lambda_value * uavs_e_tables[UAV][s][a]'''
                    # -------------------------PROVA2----------------------------------------------------------------------------

                    elif (conf.SARSA_lambda == True) and (conf.SARSA == True) or (conf.SARSA_lambda == True) and (
                            conf.Q_LEARNING == True) or (conf.SARSA == True) and (conf.Q_LEARNING == True):
                        assert False, "Invalid algorithm selection."

                if conf.SARSA_lambda == True:
                    q_values_current_episode[UAV] = uavs_q_tables[UAV][obs][action]
                    uavs_q_tables[UAV][obs][action] = uavs_q_tables[UAV][obs][action]
                else:
                    q_values_current_episode[UAV] = new_q
                    uavs_q_tables[UAV][obs][action] = new_q

                if conf.HOSP_SCENARIO == False:
                    # uavs_episode_reward[UAV] += reward
                    uavs_episode_reward[UAV] += reward
                    current_UAV_bandwidth[UAV] += conf.UAV_BANDWIDTH - agents[UAV]._bandwidth
                    current_requested_bandwidth[UAV] += env.current_requested_bandwidth

                if conf.HOSP_SCENARIO == False:
                    reset_uavs(agents[UAV])

            if done and conf.HOSP_SCENARIO == True:
                # print("uavs_episode_reward[UAV]", uavs_episode_reward[UAV], reward)

                # reset_uavs(agents[1])

                break

        if done and conf.HOSP_SCENARIO == True:
            # print("uavs_episode_reward[UAV]", uavs_episode_reward[UAV], reward)
            reset_uavs(agents[UAV])
            break
            # reset_uavs(agents[UAV])

        if conf.HOSP_SCENARIO == False:
            current_QoE3 += len(env.all_users_in_all_foots) / len(env.users) if len(
                env.users) != 0 else 0  # --> Percentage of covered users (including also the users which are not requesting a service but which are considered to have the communication with UAVs on)
            n_active_users_per_episode[episode - 1] = n_active_users

    if conf.HOSP_SCENARIO == False:
        if (conf.INF_REQUEST == False):
            tr_active_users_current_ep = tr_active_users_current_it / conf.ITERATIONS_PER_EPISODE
            ec_active_users_current_ep = ec_active_users_current_it / conf.ITERATIONS_PER_EPISODE
            dg_active_users_current_ep = dg_active_users_current_it / conf.ITERATIONS_PER_EPISODE

            for service_idx in range(N_SERVICES):
                n_users_provided_for_current_service = current_provided_services[
                                                           service_idx] / conf.ITERATIONS_PER_EPISODE

                if (service_idx == 0):
                    n_active_users_per_current_service = tr_active_users_current_ep  # tr_active_users
                elif (service_idx == 1):
                    n_active_users_per_current_service = ec_active_users_current_ep  # ec_active_users
                elif (service_idx == 2):
                    n_active_users_per_current_service = dg_active_users_current_ep  # dg_active_users

                perc_users_provided_for_current_service = n_users_provided_for_current_service / n_active_users_per_current_service if n_active_users_per_current_service != 0 else 0
                provided_services_per_epoch[episode - 1][service_idx] = perc_users_provided_for_current_service

    if (conf.UNLIMITED_BATTERY == False):
        crashes_history[episode - 1] = crashes_current_episode

    if conf.HOSP_SCENARIO == False:
        n_active_users_current_ep = n_active_users_current_it / conf.ITERATIONS_PER_EPISODE
        users_served_time = sum(QoEs_store[0]) / (len(QoEs_store[0])) if len(QoEs_store[
                                                                                 0]) != 0 else 0  # --> It is divided by its lenght, because here it is measured the percenteage according to which a service is completed (once it starts).
        users_request_service_elapsed_time = sum(QoEs_store[
                                                     1]) / n_active_users_current_ep if n_active_users_current_ep != 0 else 0  # --> It is divided by the # of active users, beacuse here it is measured the avg elapsed time (among the active users) between a service request and its provision.

        QoE3_for_current_epoch = current_QoE3 / conf.ITERATIONS_PER_EPISODE
        User.avg_QoE(episode, users_served_time, users_request_service_elapsed_time, QoE3_for_current_epoch,
                     avg_QoE1_per_epoch, avg_QoE2_per_epoch,
                     avg_QoE3_per_epoch)  # --> users_request_service_elapsed_time has to be divided by the iterations number if you do not have the mean value!!!

    print(" - Iteration: {it:1d} - Reward per UAV {uav:1d}: {uav_rew:6f}".format(it=i + 1, uav=UAV + 1, uav_rew=reward))

    for UAV in range(conf.N_UAVS):
        if conf.HOSP_SCENARIO == False:
            UAVs_used_bandwidth[UAV][episode - 1] = current_UAV_bandwidth[UAV] / conf.ITERATIONS_PER_EPISODE
            users_bandwidth_request_per_UAVfootprint[UAV][episode - 1] = current_requested_bandwidth[
                                                                             UAV] / conf.ITERATIONS_PER_EPISODE
        best_reward_episode[UAV].append(best_policy[UAV])
        current_mean_reward = uavs_episode_reward[UAV] / conf.ITERATIONS_PER_EPISODE
        uavs_episode_rewards[UAV].append(current_mean_reward)
        current_q_mean = q_values_current_episode[UAV] / conf.ITERATIONS_PER_EPISODE
        q_values[UAV].append(current_q_mean)
        print(" - Mean reward per UAV{uav:3d}: {uav_rew:6f}".format(uav=UAV + 1, uav_rew=current_mean_reward), end=" ")
    print()

    # env.render(saving_directory, episode, 1)
    # breakpoint()

    # print("\nRendering animation for episode:", episode)
    # env.render()
    # print("Animation rendered.\n")

    # if ((episode%500)==0): #(episode%250)==0 # --> You can change the condition to show (to save actually) or not the scenario of the current episode.

    if conf.HOSP_SCENARIO == False:
        env.render(saving_directory, episode, 500)

    # env.render(saving_directory, episode, 10000) # --> DO not work properly when using UAVsTemporalWrapper --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # breakpoint()

    '''if ((episode%500)==0):
        env.render()
        plot.users_wait_times(env.n_users, env.users, saving_directory, episode)'''

    # env.agents_paths = [[0 for iteration in range(conf.ITERATIONS_PER_EPISODE)] for uav in range(conf.N_UAVS)] # --> Actually is should be useless because the values are overwritten on the previous ones.
    if conf.HOSP_SCENARIO == False:
        n_discovered_users = len(env.discovered_users)

        if ((n_discovered_users / env.n_users) >= 0.85):
            EPSILON = EPSILON * EPSILON_DECREMENT if EPSILON > EPSILON_MIN else EPSILON_MIN
        else:
            EPSILON = EPSILON * EPSILON_DECREMENT if EPSILON > EPSILON_MIN else EPSILON_MIN2
        # print(fine)
    else:
        print("best_policy", best_policy[0])
        if (EPSILON_DECR == True):
            if ((best_policy[0] / 20) >= 0.95):
                EPSILON = EPSILON * EPSILON_DECREMENT if EPSILON > EPSILON_MIN else EPSILON_MIN
            else:
                EPSILON = EPSILON * EPSILON_DECREMENT if EPSILON > EPSILON_MIN else EPSILON_MIN2
        else:
            pass

    # EPSILON_MIN = 0.1
    # EPSILON_MIN2 = 0.01
    '''if ((best_policy[0]/ 20) >= 0.95):
        if EPSILON > EPSILON_MIN2:
            EPSILON = EPSILON * EPSILON_DECREMENT
        else:
            EPSILON_MIN2
    else:
        if EPSILON > EPSILON_MIN:
            EPSILON = EPSILON * EPSILON_DECREMENT
        else:
            EPSILON_MIN'''
    if (conf.REDUCE_ITERATION_PER_EPISODE == True):
        if (done and uavs_episode_reward[UAV] / 20 >= 0.95):  # best_policy[UAV] ):
            conf.ITERATIONS_PER_EPISODE = current_iteration
        if (conf.ITERATIONS_PER_EPISODE < 90):
            print("conf.ITERATIONS_PER_EPISODE:", conf.ITERATIONS_PER_EPISODE)

    if (uavs_episode_reward[UAV] > 2 and conf.ITERATIONS_PER_EPISODE > current_iteration):  # best_policy[UAV] ):
        conf.ITERATIONS_PER_EPISODE = current_iteration
        conf.EPISODES_BP = episode

    if (conf.ITERATIONS_PER_EPISODE < 90):
        print("Number of iterations (best policy):", conf.ITERATIONS_PER_EPISODE, "Episode: ", conf.EPISODES_BP)

file = open(join(saving_directory, "env_and_train_info.txt"), "a")

print("\nTRAINING COMPLETED.\n")

# breakpoint()
# ________________________________________________________________________________ Training end ________________________________________________________________________________________________________________


# ________________________________________________________________________________ Results Saving: ______________________________________________________________________________________________________

for uav_idx in range(conf.N_UAVS):
    if (conf.UNLIMITED_BATTERY == False):
        print("\nSaving battery levels when start to charge . . .")
        plot.battery_when_start_to_charge(battery_in_CS_history, uavs_directories)
        print("Battery levels when start to charge saved.")
        print("Saving UAVs crashes . . .")
        plot.UAVS_crashes(conf.EPISODES, crashes_history, saving_directory)
        print("UAVs crashes saved.")
    if conf.HOSP_SCENARIO == False:
        list_of_lists_of_actions = list(explored_states_q_tables[uav_idx].values())
        actions_values = [val for sublist in list_of_lists_of_actions for val in sublist]
        file.write("\nExploration percentage of the Q-Table for UAV:\n")
        actual_uav_id = uav_idx + 1
        value_of_interest = np.mean(actions_values)
        file.write(str(actual_uav_id) + ": " + str(value_of_interest))
        print("Exploration percentage of the Q-Table for UAV:\n", actual_uav_id, ":", value_of_interest)

    print("Saving the best policy for each UAV . . .")
    np.save(uavs_directories[uav_idx] + f"/best_policy.npy", best_policy_obs[uav_idx])
    print("Best policies saved.")

for qoe_num in range(1, 4):
    file.write("\nQoE" + str(qoe_num) + " : ")

    if (qoe_num == 1):
        file.write(str(np.mean(avg_QoE1_per_epoch)))

    if (qoe_num == 2):
        file.write(str(np.mean(avg_QoE2_per_epoch)))

    if (qoe_num == 3):
        file.write(str(np.mean(avg_QoE3_per_epoch)))

file.close()

print("\nBEST POLICY:\n")
print(len(best_policy_obs))
print(best_policy_obs)
print("\n")

# DA CAPIRE
if conf.HOSP_SCENARIO == False:
    print("\nSaving QoE charts, UAVs rewards and Q-values . . .")
    legend_labels = []
    plot.QoE_plot(avg_QoE1_per_epoch, conf.EPISODES, join(saving_directory, "QoE1"), "QoE1")
    plot.QoE_plot(avg_QoE2_per_epoch, conf.EPISODES, join(saving_directory, "QoE2"), "QoE2")
    plot.QoE_plot(avg_QoE3_per_epoch, conf.EPISODES, join(saving_directory, "QoE3"), "QoE3")

    if (conf.INF_REQUEST == False):
        plot.bandwidth_for_each_epoch(conf.EPISODES, saving_directory, UAVs_used_bandwidth,
                                      users_bandwidth_request_per_UAVfootprint)
        plot.users_covered_percentage_per_service(provided_services_per_epoch, conf.EPISODES,
                                                  join(saving_directory, "Services Provision"))

    for uav in range(conf.N_UAVS):
        # plot.UAVS_reward_plot(conf.EPISODES, best_reward_episode, saving_directory, best_reward_episodee=True)
        plot.UAVS_reward_plot(conf.EPISODES, uavs_episode_rewards, saving_directory)
        plot.UAVS_reward_plot(conf.EPISODES, q_values, saving_directory, q_values=True)

    print("Qoe charts, UAVs rewards and Q-values saved.")
    print("\nSaving Epsilon chart trend . . .")
    plot.epsilon(epsilon_history, conf.EPISODES, saving_directory)
    print("Epsilon chart trend saved.")

    print("Saving Q-Tables for episode", episode, ". . .")
    for uav in range(1, conf.N_UAVS + 1):
        saving_qtable_name = q_tables_directories[uav - 1] + f"/qtable-ep{episode}.npy"
        np.save(saving_qtable_name, uavs_q_tables)
    print("Q-Tables saved.\n")

    with open(join(saving_directory, "q_tables.pickle"), 'wb') as f:
        pickle.dump(uavs_q_tables, f)

    '''print("Saving Min and Max values related to the Q-Tables for episode", episode, ". . .")
    for uav in range(1, conf.N_UAVS+1):
        plot.actions_min_max_per_epoch(uavs_q_tables, q_tables_directories[uav-1], episode, uav)
    print("Min and Max values related to the Q-Tables for episode saved.\n")'''

policy_n.append(best_policy_obs)
print("Policy:", policy_n, '\n')

print(best_policy_obs, "best_policy_obs", len(best_policy_obs[0]))

for i in range(len(best_policy_obs[0])):  # Elimino le tuple vuote in caso ci fossero.
    if (best_policy_obs[0][i] != ()):
        if conf.HOSP_SCENARIO == True:
            traj_j.append(list(best_policy_obs[0][i][0]))
            plot_policy.append(best_policy_obs[0][i][0])
            traj_j_ID.append(list(best_policy_obs[0][i][0]))
        else:
            traj_j.append(list(best_policy_obs[0][i]))
            plot_policy.append(best_policy_obs[0][i])
            traj_j_ID.append(list(best_policy_obs[0][i]))
    else:
        break

policy_per_plot.append(plot_policy)
plot_policy = []

print("TRAJ_J:", traj_j, '\n')
print("policy_per_plot:", policy_per_plot, '\n')

traj_j_ = [traj_j]
traj = []
for i in range(len(traj_j_)):
    for k in range(len(traj_j_[i])):
        traj.append(traj_j_[i][k])
print("Old_traj_j", traj, '\n')

for i in range(len(traj_j_ID)):
    traj_j_ID[i].insert(0, j)
    traj_j_ID[i].insert(1, i)
    traj_j_ID[i].insert(4, conf.UAV_HEIGHT_Z)  # --> ?????????????????????????????????????????????????

CSV_writer("best_policy_obs.csv", best_policy_obs)  # Policy con hovering su ospedali

with open('./' + BP_DIRECTORY_NAME + "policy_per_plot_.csv", "a") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(policy_per_plot)

CSV_writer("trajectory.csv", traj)

out = open('./' + BP_DIRECTORY_NAME + 'Best_policys.csv', 'a')  # Best policy della missione con ID
for row in traj_j_ID:
    for column in row:
        out.write('%d, ' % column)
    out.write('\n')
out.close()

generate_trajectories('UAV-' + str(j) + '_BestPolicy.csv',
                      traj_j_ID)  # Best policy contenuta dentro la cartella con l' 'ID' dell'UAV

env.close()
if conf.HOSP_SCENARIO == True:
    env.reset()

if j != conf.N_MISSION-1:
    import shutil
    myfile_initial_users = "./initial_users/"
    myfile_map_data = "./map_data/"
    myfile_map_status = "./map_status/"
    myfile__pycache__ = "./__pycache__/"

    shutil.rmtree(myfile_initial_users, ignore_errors=True)
    shutil.rmtree(myfile_map_data, ignore_errors=True)
    shutil.rmtree(myfile_map_status, ignore_errors=True)
    shutil.rmtree(myfile__pycache__, ignore_errors=True)
else:
    pass


#breakpoint()

# ________________________________________________________________________________________________________________________________________________________________________________________________________


