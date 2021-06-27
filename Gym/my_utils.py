# MAIN VARIABLES DEFINITION. 

from math import ceil, pi, sin, cos
import numpy as np
from enum import Enum
import csv
from os import mkdir
from os.path import join, isdir
from datetime import datetime
from iter import j
import os
import ast

CSV_DIRECTORY_NAME = "Flights_trajectories"
BP_DIRECTORY_NAME = 'Best_policy/'
myfile = "./Best_policy/trajectory.csv"
myfile_plot = "./Best_policy/policy_per_plot_.csv"

'''with open("iter.py", "w") as text_file:
    text_file.write('j =' + '{}'.format(0))'''

if not isdir(CSV_DIRECTORY_NAME): mkdir(CSV_DIRECTORY_NAME)
if not isdir(BP_DIRECTORY_NAME): mkdir(BP_DIRECTORY_NAME)

if os.path.isfile(myfile):
	with open('./Best_policy/trajectory.csv', 'r', newline='') as file:
		myreader = csv.reader(file, delimiter='-')
		traj_csv = []
		for rows in myreader:
			traj_csv.append([int(x) for x in rows])
	#print(traj_csv)
else:
	traj_csv = []

#CONFIGURATION------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TRAINING PARAMETERS
EPISODES = 8000 # epochs
REDUCE_ITERATION_PER_EPISODE = False
ITERATIONS_PER_EPISODE = 80
NUMBER_ITER = ITERATIONS_PER_EPISODE
NO_INTERFERENCE_REWARD = False
EPISODES_BP = 0

N_MISSION = 3

hosp_pos= [[(2, 6), (6, 4)], [(5,7), (4,2)], [(9, 5), (2, 3)], [(3,9),(12,0)], [(18, 12),(19, 0)] , [(15, 6),(1, 12)], [(1, 18), (4, 15)], [(6, 17), (2, 15)] , [(17, 17), (12, 17)], [(18, 15), (10, 18)]] #Posizione ospedali
randomm = False #Posizione randomica degli ospedali
Colorss = [['None', 'gold', 'orange', 'orange'], ['None', 'gold', 'orange'], ['None', 'gold', 'orange'], ['None', 'gold', 'orange'],['None', 'gold', 'orange'], ['None', 'gold', 'orange'], ['None', 'gold', 'orange'], ['None', 'gold', 'orange'],['None', 'gold', 'orange'],['None', 'gold', 'orange']]
UAVS_POS = [[(0, 5, 0), (6, 4, 0), (1, 3, 0)], [(3, 9, 0), (4, 4, 0), (4, 4, 0)], [(9, 1, 0)]] # caso tesi
#UAVS_POS = [[(9, 0, 0)], [(3, 9, 0)], [(9, 1, 0)], [(1, 11, 0)], [(15, 15, 0)], [(14, 5, 0)] , [(4, 18, 0)], [(9, 19, 0)], [(19, 19, 0)], [(19, 14, 0)]]
UAV_HEIGHT_Z = 0
nb_colors = 2

hosp_pos_ = []
UAVS_POS_ = []
for idx in range(N_MISSION):			#Carico nello scenario solo gli ospedali e le posizioni corrispondenti alla missione.
	hosp_pos_.append(hosp_pos[idx])
	UAVS_POS_.append(UAVS_POS[idx])
hosp_pos = hosp_pos_
UAVS_POS = UAVS_POS_

assert len(hosp_pos)==len(UAVS_POS), "The number of missions must be the same as the number of UAV_POS " #Ricordare di cambiare il numero di missioni in Main.py

SHOW_PLOT = False #Mostra i goal temporali da raggiungere per ogni missione
#ok = env.o + 1


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def UAV_i(UAVS_POS, hosp_pos, Colors, j):
	UAVS_POS_ = UAVS_POS[j]
	print(UAVS_POS)
	Colors_ = Colors[j]
	return Colors_, UAVS_POS_


'''def idx_pos():
	idx = o
	return idx'''

'''ok = []
for i in range(10):
	for j in range(10):
		ok.append([i, j])
		_x_coord = i
		_y_coord = j
		matrix_elems[hosp._x_coord][hosp._y_coord]._priority = "#8b4513"

print(ok)'''

#hosp_pos = ok

def hex_to_rgb(hex):
     hex = hex.lstrip('#')
     hlen = len(hex)
     rgb = tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
     rgb_percentage = [color_part/255 for color_part in rgb]
     return tuple(rgb_percentage)

# LEARNING SELECTION:
Q_LEARNING = True
SARSA = False
SARSA_lambda = False

HOSP_SCENARIO = True

P_PURPLE = "#cf03fc"
P_GOLD = '#FFD700'
P_ORANGE = "#fc8c03"
P_BROWN = "#8b4513"

traj_j = []
policy_per_plot = []
plot_policy = []
traj_j_ID = []


P_GOLD = '#FFD700'
P_ORANGE = "#F08000"

P_BLU = "#0000FF"
P_DARK_BLU = "#000080"

P_GREEN = "#66ff99"
P_DARK_GREEN = "#026b2c"

P_FUCHSIA = "#FF00FF"
P_PURPLE = "#cf03fc"

P_PINK = "#FFB6C1"
P_DARK_PINK= "#FF69B4"

P_BROWN = "#A0522D"
P_BROWN_DARK = "#8B4513"

P_CYAN = "#7dbfc7"
P_DARK_CYAN = "#79a3b6"

P_LAVANDER = "#E6E6FA"
P_DARK_LAVANDER = "#D8BFD8"

P_NAVAJO= "#FFDEAD"
P_DARK_NAVAJO = "#F4A460"

P_GRAY = "#708090"
P_DARK_GREY = "#2F4F4F"

P_SALMON = "#FA8072"
P_DARK_SALMON = "#E9967A"

P_SKY_BLUE = "#ADD8E6"
P_DARK_SKY_BLUE= "#00BFFF"

P_LIGHT_GREEN = "#90EE90"
P_DARK_LIGHT_GREEN = "#7FFF00"

P_PAPAYA_YELLOW = "#FAFAD2"
P_DARK_PAPAYA_YELLOW = "#FFEFD5"

P_RED_LIGHT = "#DC143C"
P_RED_LIGHT_DARK = "#B22222"

P_RED = "#FF0000"
P_DARK_RED = "#8B0000"

HOSP_PRIORITIES = {1: P_GOLD, 2: P_ORANGE ,3:P_BLU, 4:P_DARK_BLU , 5:P_GREEN, 6:P_DARK_GREEN, 7:P_FUCHSIA, 8:P_PURPLE,
				   9: P_PINK, 10: P_DARK_PINK ,11:P_BROWN, 12:P_BROWN_DARK , 13:P_CYAN, 14:P_DARK_CYAN, 15:P_LAVANDER, 16:P_DARK_LAVANDER,
				   17: P_NAVAJO, 18: P_DARK_NAVAJO ,19:P_GRAY, 20:P_DARK_GREY , 21:P_SALMON, 22:P_DARK_SALMON, 23:P_SKY_BLUE, 24:P_DARK_SKY_BLUE,
				   25: P_LIGHT_GREEN, 26: P_DARK_LIGHT_GREEN ,27:P_PAPAYA_YELLOW, 28:P_DARK_PAPAYA_YELLOW , 29:P_RED_LIGHT, 30:P_RED_LIGHT_DARK}
PRIORITY_NUM = len(HOSP_PRIORITIES)

def get_color_name(color):

	if (color==P_GOLD):
		color_name = "gold"
	elif (color==P_ORANGE):
		color_name = "orange"
	elif (color == P_BLU):
		color_name = "blu"
	elif (color == P_DARK_BLU):
		color_name = "dark_blu"
	elif (color == P_GREEN):
		color_name = "green"
	elif (color == P_DARK_GREEN):
		color_name = "dark_green"
	elif (color == P_FUCHSIA):
		color_name = "fuchsia"
	elif (color==P_PURPLE):
		color_name = "purple"
	elif (color == P_PINK):
		color_name = "pink"
	elif (color==P_DARK_PINK):
		color_name = "dark_pink"
	elif (color == P_BROWN):
		color_name = "brown"
	elif (color == P_BROWN_DARK):
		color_name = "brown_dark"
	elif (color == P_CYAN):
		color_name = "cyan"
	elif (color == P_DARK_CYAN):
		color_name = "dark_cyan"
	elif (color == P_LAVANDER):
		color_name = "lavander"
	elif (color == P_DARK_LAVANDER):
		color_name = "dark_lavander"

	elif (color == P_RED):
		color_name = "red"
	else:
		color_name = None

	return color_name

def get_color_id(color):
	#print("GUARDA QUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA: ", color)

	if (color==P_GOLD):
		color_id = 1
	elif (color==P_ORANGE):
		color_id = 2
	elif (color == P_BLU):
		color_id = 3
	elif (color == P_DARK_BLU):
		color_id = 4
	elif (color == P_GREEN):
		color_id = 5
	elif (color == P_DARK_GREEN):
		color_id = 6
	elif (color == P_FUCHSIA):
		color_id = 7
	elif (color==P_PURPLE):
		color_id = 8
	elif (color == P_PINK):
		color_id = 9
	elif (color==P_DARK_PINK):
		color_id = 10
	elif (color == P_BROWN):
		color_id = 11
	elif (color == P_BROWN_DARK):
		color_id = 12
	elif (color == P_CYAN):
		color_id = 13
	elif (color == P_DARK_CYAN):
		color_id = 14
	elif (color == P_LAVANDER):
		color_id = 15
	elif (color == P_DARK_LAVANDER):
		color_id = 16
	else:
		color_id = 0


	return color_id

UAV_STANDARD_BEHAVIOUR = False

N_TIMESLOTS_PER_HOUR = 60
N_TIMESLOTS_PER_DAY = N_TIMESLOTS_PER_HOUR*24 



'''def iteration_rest():
	ITERATIONS_PER_EPISODE = 80
	return ITERATIONS_PER_EPISODE
ITERATIONS_PER_EPISODE = iteration_rest()'''
#ITERATIONS_PER_EPISODE = 80
MOVE_USERS_EACH_N_EPOCHS = 300
UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS = 10
STEPS = N_TIMESLOTS_PER_DAY


#FULL_BATTERY_LEVEL = 100
#BATTERY_LEVEL_CONSIDERED_FULL = FULL_BATTERY_LEVEL-1

CONTINUOUS_TIME = True

MIDDLE_CELL_ASSUMPTION = True

N_SERVICES = 3

# OBSERVATION
N_UC = 3
N_CS = 2


if j != None:  #Aggiustare in un modo più elegante.. Se j = None, plot dello scenario finale.
	N_HOSP = len(hosp_pos[j])
else:
	N_HOSP = 2





count = 0
for i in range(len(hosp_pos)):
	for j in range(len(hosp_pos[i])):
		count +=1
N_HOSP_TOT = count

assert N_HOSP%2==0, "ODD HOSPITALS' NUMBER: you have to set an even number of hospitals in order to asociate each of it with another one."
assert N_HOSP/PRIORITY_NUM%2 != 0, "PRIORITIES OR HOSPITALS NUMBER RESET NEEDED: you have to set a number of priorities which is the half of the number of hospitals."
N_BATTERY_LEVELS = 10
PERC_CONSUMPTION_PER_ITERATION = 4 # --> in this way it is possible to have integer battery levels withouth approximation errors which lead to 'Key Error' inside Q-table dictionary
FULL_BATTERY_LEVEL = ITERATIONS_PER_EPISODE*PERC_CONSUMPTION_PER_ITERATION # = iteration * 4
PERC_BATTERY_CHARGED_PER_IT = 20
BATTERY_CHARGED_PER_IT = round((PERC_BATTERY_CHARGED_PER_IT*FULL_BATTERY_LEVEL)/100)
#PERC_CONSUMPTION_PER_ITERATION = STEP_REDUCTION_TO_GO_TO_CS
PERC_BATTERY_TO_GO_TO_CS = int(PERC_CONSUMPTION_PER_ITERATION/4) # --> 4/4 = 1
N_UAVS = 1
PERC_CONSUMPTION_IF_HOVERING = 2

PROPULSION_BATTERY_CONSUMPTION = 3.33
HOVERING_ENERGY_CONSUMPTION = 2
EDGE_COMPUTING_ENERGY_CONSUMPTION = 1

PERC_CRITICAL_BATTERY_LEVEL = 20
CRITICAL_BATTERY_LEVEL = round((PERC_CRITICAL_BATTERY_LEVEL*FULL_BATTERY_LEVEL)/100)
CRITICAL_BATTERY_LEVEL_2 = round(CRITICAL_BATTERY_LEVEL/1.5)
CRITICAL_BATTERY_LEVEL_3 = round(CRITICAL_BATTERY_LEVEL_2/1.5)
CRITICAL_BATTERY_LEVEL_4 = round(CRITICAL_BATTERY_LEVEL_3/1.5)
PERC_MINIMUM_BATTERY_LEVEL_TO_STOP_CHARGING = 80
MINIMUM_BATTERY_LEVEL_TO_STOP_CHARGING = round((PERC_MINIMUM_BATTERY_LEVEL_TO_STOP_CHARGING*FULL_BATTERY_LEVEL)/100)

SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT = 400

CRITICAL_WAITING_TIME_FOR_SERVICE = 10
NORMALIZATION_FACTOR_WAITING_TIME_FOR_SERVIE = 120

# AREA DIMENSIONS:


'''def reduce_map(traj_csv):
	x_max = 10
	y_max = 10
	for i in range(len(traj_csv)):
		if traj_csv[i][0] > x_max:
			x_max = traj_csv[i][0]
		else:
			pass
		if traj_csv[i][1] > y_max:
			y_max = traj_csv[i][1]
	x_max +1 , y_max +1
	if x_max > y_max: #Fino a quando non si potrà impostare un ambiente rettangolare
		y_max = x_max
	else:
		x_max = y_max
	return x_max, y_max'''

AREA_WIDTH = 12
AREA_HEIGHT = 10
LOWER_BOUNDS = 0
CELL_RESOLUTION_PER_ROW = 1
CELL_RESOLUTION_PER_COL = 1
CELLS_ROWS = int(AREA_HEIGHT/CELL_RESOLUTION_PER_ROW)
CELLS_COLS = int(AREA_WIDTH/CELL_RESOLUTION_PER_COL)
N_CELLS = CELLS_ROWS*CELLS_COLS
MINIMUM_AREA_HEIGHT = 0
MAXIMUM_AREA_HEIGHT = 12
MAX_HEIGHT_PER_USER = 3
MIN_UAV_HEIGHT = 4

# Setting RADIAL_DISTANCE_X equal to RADIAL_DISTANCE_Y, CSs will be set at an equal distance w.r.t. the eNB (eNodeB) position;
# Obviously the values of RADIAL_DISTANCE_X and RADIAL_DISTANCE_Y must not exceed the distance between the eNodeB and the area dimensions.
RADIAL_DISTANCE_X = 4
RADIAL_DISTANCE_Y = 4




CREATE_ENODEB = False
DIMENSION_2D = True # --> Enable/Disable 2D environment
UNLIMITED_BATTERY = True # --> Enable/Disable battery limitation on UAVs
INF_REQUEST = True # --> Enable/Disable continuous users requests
STATIC_REQUEST = True # --> Enable/Disable static (i.e., not moving users) request
MULTI_SERVICE = False # --> Enable/Disable limitation on UAV bandwidth
NOISE_ON_POS_MEASURE = False # --> Enable/Disable random error on the position of each UAV used 
USERS_PRIORITY = False

RAD_BETWEEN_POINTS = 2*pi/N_UAVS

if HOSP_SCENARIO == True:
	assert UNLIMITED_BATTERY == True, "UNLIMITED_BATTERY must be set to True if HOSP_SCENARIO is True"


#UAVS_POS = []
if HOSP_SCENARIO == False:
	for uav_idx in range(N_UAVS):
		if (DIMENSION_2D == True):
			UAVS_POS.append( (round(ceil(CELLS_COLS/2) + RADIAL_DISTANCE_X*sin(uav_idx*RAD_BETWEEN_POINTS)), round(ceil(CELLS_ROWS/2) + RADIAL_DISTANCE_Y*cos(uav_idx*RAD_BETWEEN_POINTS)), 0) )
		else:
			UAVS_POS.append( (round(ceil(CELLS_COLS/2) + RADIAL_DISTANCE_X*sin(uav_idx*RAD_BETWEEN_POINTS)), round(ceil(CELLS_ROWS/2) + RADIAL_DISTANCE_Y*cos(uav_idx*RAD_BETWEEN_POINTS)), MIN_UAV_HEIGHT) )

	#UAVS_POS_2D = [(4,4), (2,4)]
	#UAVS_POS =[(3, 3, 0)] # PER SCEGLIERE LA POSIZIONE DELL'UAV


TR_BATTERY_CONSUMPTION = 1
EC_BATTERY_CONSUMPTION = 2

R_MAX = False
PRIOR_KNOWLEDGE = False
X_SPLIT = 2
Y_SPLIT = 3
N_SUBAREAS = X_SPLIT*Y_SPLIT
#USERS_SERVED_TIME = 0
#USERS_REQUEST_SERVICE_ELAPSED_TIME = 0

TIME_SLOT_FOR_DELAYED_START = 5
DELAYED_START_PER_UAV = ITERATIONS_PER_EPISODE/TIME_SLOT_FOR_DELAYED_START # 30/5 = 6
UAV_FOOTPRINT = 2.5
ACTUAL_UAV_FOOTPRINT = UAV_FOOTPRINT/CELL_RESOLUTION_PER_COL

# CAMBIA IL CONSUMO DEGLI UAV NEL CASO IN CUI UTILIZZO UNA RISOLUZIONE DIVERSA DA QUELLA MINIMA --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
HALF_RESOLUTION_X =(AREA_WIDTH//CELLS_COLS)/2
HALF_RESOLUTION_Y =(AREA_WIDTH//CELLS_ROWS)/2
HALF_RESOLUTION_Z = 0.5 # --> the resolution along z is fixed
UAV_XY_STEP = 1
UAV_Z_STEP = 3

# OBSTACLES PARAMETERS:
if HOSP_SCENARIO == True:
	MIN_OBS_PER_AREA = 0.5
	MAX_OBS_PER_AREA = 1
	MIN_OBS_HEIGHT = 3
	MAX_OBS_HEIGHT = MAXIMUM_AREA_HEIGHT
else:
	MIN_OBS_PER_AREA = 0.05
	MAX_OBS_PER_AREA = 0.15
	MIN_OBS_HEIGHT = 3
	MAX_OBS_HEIGHT = MAXIMUM_AREA_HEIGHT

# CHARGING STATION PARAMETERS:
CS_HEIGHT = MIN_UAV_HEIGHT

# ENODEB POSITION:
ENODEB_X = ceil(AREA_WIDTH/2) 
ENODEB_Y = ceil(AREA_HEIGHT/2)
ENODEB_Z = 6 if DIMENSION_2D==True else 0

# CELLS STATES:
FREE = 0
OBS_IN = 1
CS_IN = 2
ENB_IN = 3
UAV_IN = 4
HOSP_IN = 5
HOSP_AND_CS_IN = 6
HOSP_AND_ENB_IN = 7

NEEDED_SERVICE_TIMES_PER_USER = [1, 2, 3, 4] 


# AGENT STATES:
UC_DISTANCES = 2 # Distances of the uncovered clusters of users 
CS_DISTANCE = 1 # Distance of the nearest Charging Station
PROCESSED_DATA = 4 
# Tuple Agent State --> (BATTERY_LEVEL, UC_DISTANCES, CS_DISTANCE, PROCESSED_DATA)

# ACTUAL AGENT STATE:
# (CELL_STATE, AGENT_STATE)

# AGENT PARAMETERS:
MAX_SPEED = 8.3 # [m/s]
MAX_ACCELERATION = 4 # [m/s^2]
HOVERING_TIME = 60 # [s]
BATTERY_AUTONOMY_TIME = 1800 # [s]
MINIMUM_CHARGING_TIME = 300 # [s]
MAX_BW_PER_UAV = 5 # [MHz]

# ACTIONS:
MOVE = 0
HOVERING = 4
UP = 5
DOWN = 6
LEFT = 7
RIGHT = 8
RISE = 9
DROP = 10
MOVEMENTS = [UP, DOWN, LEFT, RIGHT, RISE, DROP]
GO_TO_CS = 1
CHARGE = 2
PROCESS_DATA = 3
NOT_ALLOWED_MOTION = -1


AGENTS_PATHS = [[] for uav in range(N_UAVS)]


ACTION_SPACE_3D_MIN = [UP, DOWN, LEFT, RIGHT, RISE, DROP, HOVERING] # TO DEFINE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ACTION_SPACE_3D_COME_HOME = [UP, DOWN, LEFT, RIGHT, RISE, DROP, HOVERING, GO_TO_CS]
ACTION_SPACE_3D_WHILE_CHARGING = [UP, DOWN, LEFT, RIGHT, RISE, DROP, HOVERING, CHARGE]
ACTION_SPACE_3D_TOTAL = [UP, DOWN, LEFT, RIGHT, RISE, DROP, HOVERING, GO_TO_CS, CHARGE]
GO_TO_CS_3D_INDEX = ACTION_SPACE_3D_TOTAL.index(GO_TO_CS)
GO_TO_CS_3D_INDEX_HOME_SPACE = ACTION_SPACE_3D_COME_HOME.index(GO_TO_CS)
CHARGE_3D_INDEX = ACTION_SPACE_3D_TOTAL.index(CHARGE)
CHARGE_3D_INDEX_WHILE_CHARGING = ACTION_SPACE_3D_WHILE_CHARGING.index(CHARGE)

ACTION_SPACE_2D_MIN = [UP, DOWN, LEFT, RIGHT, HOVERING]
ACTION_SPACE_2D_COME_HOME = [UP, DOWN, LEFT, RIGHT, HOVERING, GO_TO_CS]
ACTION_SPACE_2D_WHILE_CHARGING = [UP, DOWN, LEFT, RIGHT, HOVERING, CHARGE]
ACTION_SPACE_2D_TOTAL = [UP, DOWN, LEFT, RIGHT, HOVERING, GO_TO_CS, CHARGE]
GO_TO_CS_2D_INDEX = ACTION_SPACE_2D_TOTAL.index(GO_TO_CS)
GO_TO_CS_2D_INDEX_HOME_SPACE = ACTION_SPACE_2D_COME_HOME.index(GO_TO_CS)
CHARGE_2D_INDEX = ACTION_SPACE_2D_TOTAL.index(CHARGE)
CHARGE_2D_INDEX_WHILE_CHARGING = ACTION_SPACE_2D_WHILE_CHARGING.index(CHARGE)

if (DIMENSION_2D==True):
	if (UNLIMITED_BATTERY == True):
		DICT_ACTION_SPACE_2D = {ACTION_SPACE_2D_MIN.index(UP): "UP", ACTION_SPACE_2D_MIN.index(DOWN): "DOWN", ACTION_SPACE_2D_MIN.index(LEFT): "LEFT", ACTION_SPACE_2D_MIN.index(RIGHT): "RIGHT", ACTION_SPACE_2D_MIN.index(HOVERING): "HOVER"}
	else:
		DICT_ACTION_SPACE_2D = {ACTION_SPACE_2D_TOTAL.index(UP): "UP", ACTION_SPACE_2D_TOTAL.index(DOWN): "DOWN", ACTION_SPACE_2D_TOTAL.index(LEFT): "LEFT", ACTION_SPACE_2D_TOTAL.index(RIGHT): "RIGHT", ACTION_SPACE_2D_TOTAL.index(HOVERING): "HOVER", ACTION_SPACE_2D_TOTAL.index(GO_TO_CS): "GO CS", ACTION_SPACE_2D_TOTAL.index(CHARGE): "CHARGE"}	
else:
	if (UNLIMITED_BATTERY == True):
		DICT_ACTION_SPACE_3D = {ACTION_SPACE_3D_MIN.index(UP): "UP", ACTION_SPACE_3D_MIN.index(DOWN): "DOWN", ACTION_SPACE_3D_MIN.index(LEFT): "LEFT", ACTION_SPACE_3D_MIN.index(RIGHT): "RIGHT", ACTION_SPACE_3D_MIN.index(RISE): "RISE", ACTION_SPACE_3D_MIN.index(DROP): "DROP", ACTION_SPACE_3D_MIN.index(HOVERING): "HOVER"}
	else:
		DICT_ACTION_SPACE_3D = {ACTION_SPACE_3D_TOTAL.index(UP): "UP", ACTION_SPACE_3D_TOTAL.index(DOWN): "DOWN", ACTION_SPACE_3D_TOTAL.index(LEFT): "LEFT", ACTION_SPACE_3D_TOTAL.index(RIGHT): "RIGHT", ACTION_SPACE_3D_TOTAL.index(RISE): "RISE", ACTION_SPACE_3D_TOTAL.index(DROP): "DROP", ACTION_SPACE_3D_TOTAL.index(HOVERING): "HOVER", ACTION_SPACE_3D_TOTAL.index(GO_TO_CS): "GO CS", ACTION_SPACE_3D_TOTAL.index(CHARGE): "CHARGE"}

DICT_ACTION_SPACE = DICT_ACTION_SPACE_2D if DIMENSION_2D==True else DICT_ACTION_SPACE_3D
LIMIT_VALUES_FOR_ACTION = [] # --> sarà una lista di tuple (min_val, max_val)

ACTION_SPACE_STANDARD_BEHAVIOUR = [UP, DOWN, LEFT, RIGHT, RISE, GO_TO_CS, CHARGE]

# ENVIRONMENT PARAMETERS:
BS_POS = np.vstack((ENODEB_X, ENODEB_Y, ENODEB_Z))
AVG_BUILDS_HEIGHT = 4
AVG_ROAD_WIDTH = 20
AVG_BUILDS_SEPARATION = 36
PHI = 10 # Slope roads angle
OPERATING_FREQUENCY = 2.1
BS_BANDWIDTH = 40 # [MHz]
UAV_BANDWIDTH = 5000 # [KHz] più o meno [Kbps]

# MODULATION:
NoTx = 0
QPSK = 1
_16QAM = 2
_64QAM = 3

# LOOKUP TABLE:
LOOKUP_TABLE = {'CQI_index': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 , 14, 15],
				'modulation': [NoTx, NoTx, NoTx, QPSK, QPSK, QPSK, _16QAM, _16QAM, _16QAM, _64QAM, _64QAM, _64QAM, _64QAM, _64QAM, _64QAM],
				'code_rate': [None, None, None, 0.3, 0.44, 0.59, 0.37, 0.48, 0.6, 0.45, 0.55, 0.65, 0.75, 0.85, 0.93],
				'bits_per_symbol': [None, None, None, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547],
				'SINR_dB': [-1.25, -1.25, -1.25, -0.94, 1.09, 2.97, 5.31, 6.72, 8.75, 10.47, 12.34, 14,37, 15.94, 17.81, 20.31],
				'TBS': [0, 0, 0, 384, 576, 768, 960, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840],
				'SE': [0.1065, 0.1640, 0.2638, 0.4211, 0.6139, 0.8222, 1.0333, 1.3389, 1.6833, 1.9111, 2.3222, 2.7278, 3.1611, 3.5778, 3.8833]
				}

# MIN AND MAX VALUES TO DETERMINE THE EVENTUAL MAX PEAK OF USER PER DAY:
MIN_USERS_PER_DAY = 15
MAX_USERS_PER_DAY = 30 

# MAX AND MIN PERCENTAGE NUMBER OF USERS PER TIMESLOT:
MIN_USERS_PERCENTAGE1 = 0.15 # Timeslot 1-4 A.M. 
MIN_USERS_PERCENTAGE2 = 0.30 # Timeslot 4-8 A.M.
MIN_USERS_PERCENTAGE3 = 0.40 # Timeslot 8-12 A.M.
MIN_USERS_PERCENTAGE4 = 0.50 # Timeslot 1-4 P.M.
MIN_USERS_PERCENTAGE5 = 0.80 # Timeslot 4-8 P.M.
MIN_USERS_PERCENTAGE6 = 0.30 # Timeslot 8-12 P.M.

MAX_USERS_PERCENTAGE1 = 0.25 # Timeslot 1-4 A.M.
MAX_USERS_PERCENTAGE2 = 0.40 # Timeslot 4-8 A.M.
MAX_USERS_PERCENTAGE3 = 0.50 # Timeslot 8-12 A.M.
MAX_USERS_PERCENTAGE4 = 0.70 # Timeslot 1-4 P.M.
MAX_USERS_PERCENTAGE5 = 1.0 # Timeslot 4-8 P.M.
MAX_USERS_PERCENTAGE6 = 0.45 # Timeslot 8-12 P.M.

MIN_MAX_USERS_PERCENTAGES = [(MIN_USERS_PERCENTAGE1, MAX_USERS_PERCENTAGE1), (MIN_USERS_PERCENTAGE2, MAX_USERS_PERCENTAGE2), \
							 (MIN_USERS_PERCENTAGE3, MAX_USERS_PERCENTAGE3), (MIN_USERS_PERCENTAGE4, MAX_USERS_PERCENTAGE4), \
							 (MIN_USERS_PERCENTAGE5, MAX_USERS_PERCENTAGE5), (MIN_USERS_PERCENTAGE6, MAX_USERS_PERCENTAGE6)]

MACRO_TIME_SLOTS = 6
HOURS_PER_CONSIDERED_TIME = 24
MICRO_SLOTS_PER_MACRO_TIMESLOT = HOURS_PER_CONSIDERED_TIME//MACRO_TIME_SLOTS
STARTING_TIMESLOT = MICRO_SLOTS_PER_MACRO_TIMESLOT 
HOURS_VALID_UPPER_BOUND = HOURS_PER_CONSIDERED_TIME+1

# -------------------------------------------------------------- USERS TYPE OF REQUESTS --------------------------------------------------------------

# USERS BITRATE REQUESTS [Kb/s]:
AUDIO_STREAM_NORMAL = 96
AUDIO_STREAM_HIGH = 160
AUDIO_STREAM_EXTREME = 320
VIDEOCALL = 200
YOUTUBE_360p = 750
YOUTUBE_480p = 1000
YOUTUBE_720p = 2500
TRHOUGHPUT_REQUESTS = [AUDIO_STREAM_NORMAL, VIDEOCALL, AUDIO_STREAM_HIGH, YOUTUBE_360p, AUDIO_STREAM_EXTREME, YOUTUBE_480p, YOUTUBE_720p]

TR_SERVICE_TIME1 = 1
TR_SERVICE_TIME2 = 2
TR_SERVICE_TIME3 = 3
TR_SERVICE_TIME4 = 4
TR_SERVICE_TIME5 = 5
TR_SERVICE_TIME6 = 6
TR_SERVICE_TIMES = [TR_SERVICE_TIME1, TR_SERVICE_TIME2, TR_SERVICE_TIME3, TR_SERVICE_TIME4, TR_SERVICE_TIME5, TR_SERVICE_TIME6]

# USERS BYTE REQUEST FOR EDGE-COMPUTING:
BYTE1 = 10
BYTE2 = 20
BYTE3 = 30
BYTE4 = 40
BYTE5 = 50
EDGE_COMPUTING_REQUESTS = [BYTE1, BYTE2, BYTE3, BYTE4, BYTE5]

EC_SERVICE_TIME = 3

# USERS MESSAGES FOR DATA GATHERING:
MESSAGE1 = 3
MESSAGE2 = 6
MESSAGE3 = 9
MESSAGE4 = 12
MESSAGE5 = 24
DATA_GATHERING_REQUESTS = [MESSAGE1*8, MESSAGE2*8, MESSAGE3*8, MESSAGE4*8, MESSAGE5*8]

DG_SERVICE_TIME = 2

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# POSSIBLE SERVICES:
NO_SERVICE = 0
THROUGHPUT_REQUEST = 1
EDGE_COMPUTING = 2
DATA_GATHERING = 3
UAVS_SERVICES = [NO_SERVICE, THROUGHPUT_REQUEST, EDGE_COMPUTING, DATA_GATHERING]
SERVICE_PROBABILITIES = [0.1, 0.5, 0.25, 0.15]

#CENTROIDS_MIN_MAX_COORDS = [[(2,2), (5, 5)], [(7, 7), (7,7)], [(6, 6), (2,2)], [(2, 2), (2,2)]] # --> [(min_x, max_x), (min_y, max_y)]
#CENTROIDS_MIN_MAX_COORDS = [[(7, 7), (7,7)], [(6, 6), (2,2)], [(2, 2), (2,2)]] # [[(7, 7), (7,7)], [(2, 2), (2,2)]]
CENTROIDS_MIN_MAX_COORDS = [[(2,2), (5, 5)]] #, [(7, 7), (7,7)]]
#CENTROIDS_MIN_MAX_COORDS = [[(2,2), (5, 5)]]
FIXED_CLUSTERS_NUM = len(CENTROIDS_MIN_MAX_COORDS)
# LIST OF THE NUMBER OF CLUSTERS TO TEST WHEN LOOKING FOR THE OPTIMAL NUMBER OF USERS CLUSTERS:
CLUSTERS_NUM_TO_TEST = [3, 4, 5, 6, 7, 8, 9, 10]

# USER PARAMETERS:
BASE_USER = 1
FREE_USER = 0
FULL_USER = 2
PREMIUM_USER = 3
USERS_ACCOUNTS = [FREE_USER, BASE_USER, FULL_USER, PREMIUM_USER]
USERS_ACCOUNTS_DITRIBUTIONS = [0.4, 0.3, 0.2, 0.1]

# COLORS FOR PLOTTING:
WHITE = "#ffffff"
LIGHT_RED = "#ff0000"
DARK_RED = "#800000"
LIGHT_BLUE = "#66ffff"
DARK_BLUE = "#000099"
LIGHT_GREEN = "#66ff99"
DARK_GREEN = "#006600"
PURPLE = "#cf03fc"
ORANGE = "#fc8c03"
BROWN = "#8b4513"
GOLD = '#FFD700'
HEX_CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'] # Used to generate random hex color related to the users clusters.
HEX_COLOR_CODE = '#'
# UAVS colors:
VIOLET = '#EE82EE'
ORANGE = '#FFA500'
GREY = '#808080'
BROWN = '#A52A2A'
UAVS_COLORS = [P_RED_LIGHT, ORANGE, P_BLU, P_DARK_GREEN, VIOLET, BROWN, GREY, P_SALMON, P_DARK_LAVANDER, P_CYAN, P_DARK_PAPAYA_YELLOW, P_PINK, P_DARK_SKY_BLUE, P_DARK_PINK, P_FUCHSIA]
UAVS_COLORS_RGB_PERCENTAGE = [hex_to_rgb(color) for color in UAVS_COLORS]

# DIRECTORIES:
MAP_DATA_DIR = "./map_data/"
MAP_STATUS_DIR = "./map_status/"
INITIAL_USERS_DIR = "./initial_users/"
AGENTS_DIR = "./scenario_agents"

# FILES:
OBS_POINTS_LIST = "obs_points.npy"
POINTS_MATRIX = "points_matrix.npy"
CS_POINTS = "cs_points.npy"
ENODEB_POINT = "eNB_point.npy"
HOSP_POINTS = "hosp_points.npy"
PRIORITIES_POINTS = "priorities_points.npy"

CELLS_MATRIX = "cells_matrix.npy"
OBS_CELLS = "obs_cells.npy"
CS_CELLS = "cs_cells.npy"
ENB_CELLS = "enb_cells.npy"
HOSP_CELLS = "hosp_cells.npy"
PRIORITIES_CELLS = "priorities_cells.npy"

POINTS_STATUS_MATRIX = "points_status_matrix.npy"
CELLS_STATUS_MATRIX = "cells_status_matrix.npy"
PERCEIVED_STATUS_MATRIX = "perceived_status_matrix.npy"

INITIAL_USERS = "initial_users.npy"
INITIAL_CENTROIDS = "initial_centroids.npy"
INITIAL_CLUSTERS_RADIUSES = "initial_clusters_radiuses.npy"
INITIAL_CLUSTERER = "initial_clusterer.npy"
INITIAL_USERS_CLUSTERS = "initial_users_clusters.npy"

AGENTS = "agents.npy"

#CSV--------------------------------------------------------------------------------------------------------------------
dir_mission = "/Mission_"+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
def generate_single_trajectory(csv_filename, csv_dir, data):

    out = open('./' + csv_dir + '/' + csv_filename, 'a')
    #print(data, "data")
    for row in data:
        for column in row:
            out.write('%d, ' % column)
        out.write('\n')
    out.close()

num_trajectories = 1

def generate_trajectories(name, data):
    csv_dir = str(CSV_DIRECTORY_NAME+"/MissionStart"+ '_' +str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))\
         .replace("-", "_")\
        .replace(":", "_")\
        .replace(".", "")\
        .replace(" ", "--").strip()
        )
    mkdir(csv_dir)
    for i in range(0,num_trajectories):
        csv_filename=csv_dir+"/"+"flight_"+str(i+1)
        generate_single_trajectory(name, csv_dir, data)
    return None

def CSV_writer(csv_filename, data):
	with open('./' + BP_DIRECTORY_NAME + csv_filename, 'a', newline='') as file:
		mywriter = csv.writer(file, delimiter='-')
		mywriter.writerows(data)

def policy_per_plot_():
	with open('Best_policy/policy_per_plot_.csv', 'r', newline='') as file:
		myreader = csv.reader(file, delimiter=',')
		buffer = []
		policy_plot = []
		for rows in myreader:
			# ok.append([int(x) for x in rows])
			for i in range(len(rows)):
				#print(rows[i])
				buffer.append(ast.literal_eval(rows[i]))
			policy_plot.append(list(buffer))
			buffer = []
		#print(policy_plot, "policy_plot")
		'''if policy_plot == []:
			pass
		else:
			print(policy_plot)
			del policy_plot[-1]'''
	return policy_plot

if os.path.isfile(myfile_plot):
	agents_paths = policy_per_plot_()
	print("agents_paths", agents_paths)
else:
	pass
