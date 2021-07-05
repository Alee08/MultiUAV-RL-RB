import sys
from configparser import ConfigParser
import ast
from math import ceil, pi, sin, cos
#from iter import j

with open("iter.txt", "r") as text_file:
    j = int(text_file.read())


class Config:

    def __init__(self, scenario_ini='settings/scenario.ini', UAVs_ini='settings/UAVs.ini', training_ini='settings/training.ini', users_ini='settings/users.ini', hospitals_ini='settings/hospitals.ini'):
        config = ConfigParser()



        self.scenario_ini = scenario_ini
        self.UAVs_ini = UAVs_ini
        self.training_ini = training_ini
        self.users_ini = users_ini
        scenario_file = config.read(self.scenario_ini)
        UAVs_file = config.read(self.UAVs_ini)
        training_file = config.read(self.training_ini)
        users_file = config.read(self.users_ini)

        if scenario_file==[] or UAVs_file==[] or training_file==[] or users_file==[]:
            print("ERROR cannot load settings init file")
            sys.exit(1)

        # [Scenario]

        self.section = config['Scenario']

        self.HOSP_SCENARIO = True if self.section['SCENARIO_TYPE']=='hospitals' else False
        self.AREA_WIDTH = int(self.section['X'])
        self.AREA_HEIGHT = int(self.section['Y'])
        self.MAXIMUM_AREA_HEIGHT = int(self.section['Z'])
        self.CS_HEIGHT = int(self.section['CS_HEIGHT'])
        self.LOWER_BOUNDS = 0
        self.CELL_RESOLUTION_PER_COL = int(self.section['X_CELL_RESOLUTION'])
        self.CELL_RESOLUTION_PER_ROW = int(self.section['Y_CELL_RESOLUTION'])
        self.DIMENSION_2D = True if self.section['DIMENSION']=='2D' else False
        self.MIN_OBS_PER_AREA = float(self.section['MIN_OBS_PER_AREA'])
        self.MAX_OBS_PER_AREA = float(self.section['MAX_OBS_PER_AREA'])
        self.MIN_OBS_HEIGHT = float(self.section['MIN_OBS_HEIGHT'])
        self.MAX_OBS_HEIGHT = float(self.section['MAX_OBS_HEIGHT'])
        self.N_CS = int(self.section['N_CS'])
        self.CELLS_ROWS = int(self.AREA_HEIGHT/self.CELL_RESOLUTION_PER_ROW)
        self.CELLS_COLS = int(self.AREA_WIDTH/self.CELL_RESOLUTION_PER_COL)
        self.N_CELLS = self.CELLS_ROWS*self.CELLS_COLS
        # Create EnodeB in the middle of the generated map (not fully implemented yet, thus it will be not considered during the training phase):
        self.CREATE_ENODEB = False if self.section['CREATE_ENODEB'] =='False' else True
        # EnodeB location (if EnodeB is not created, its location will be used as a reference for the middle of the generated map):
        self.ENODEB_X = ceil(self.AREA_WIDTH/2) 
        self.ENODEB_Y = ceil(self.AREA_HEIGHT/2)
        self.ENODEB_Z = 6 if self.DIMENSION_2D==True else 0
        # Cells states:
        self.FREE = 0
        self.OBS_IN = 1
        self.CS_IN = 2
        self.ENB_IN = 3
        self.UAV_IN = 4
        self.HOSP_IN = 5
        self.HOSP_AND_CS_IN = 6 # Not used ?????????
        self.HOSP_AND_ENB_IN = 7 # Not used ?????????
        # Setting RADIAL_DISTANCE_X equal to RADIAL_DISTANCE_Y, CSs will be set at an equal distance w.r.t. the eNB position (and to the middle of the map);
        # Obviously the values of RADIAL_DISTANCE_X and RADIAL_DISTANCE_Y must not exceed the distance between the eNodeB and the area dimensions.
        self.RADIAL_DISTANCE_X = self.CELLS_COLS/2-1 #4
        self.RADIAL_DISTANCE_Y = self.CELLS_ROWS/2-1 #4

        # [Training]

        self.section = config['Training']

        self.Q_LEARNING = False
        self.SARSA = False
        self.SARSA_lambda = False
        self.UAV_STANDARD_BEHAVIOUR = False
        self.algorithm = None
        self.R_MAX = False
        self.PRIOR_KNOWLEDGE = False

        if self.section['ALGORITHM']=='q_learning':
            self.Q_LEARNING = True
            self.algorithm = 'Q-Learning'
        elif self.section['ALGORITHM']=='baseline':
            self.UAV_STANDARD_BEHAVIOUR = False
            self.algorithm = 'Baseline'
        elif self.section['ALGORITHM']=='sarsa':
            self.SARSA = True
            self.algorithm = 'Sarsa'
        elif self.section['ALGORITHM']=='sarsa_lambda':
            self.SARSA_lambda = True
            self.algorithm = 'SARSA Lambda'
        else:
            self.algorithm = 'invalid'
        self.EPISODES = int(self.section['EPISODES'])
        self.REDUCE_ITERATION_PER_EPISODE = True if self.section['REDUCE_ITERATION_PER_EPISODE']=='True' else False
        self.ITERATIONS_PER_EPISODE = int(self.section['ITERATIONS_PER_EPISODE'])
        self.NO_INTERFERENCE_REWARD = True if self.section['NO_INTERFERENCE_REWARD']=='True' else False
        self.EPISODES_BP = 0
        self.Q_INIT_TYPE = self.section['Q_INIT_TYPE']

        if self.Q_INIT_TYPE=='max_reward':
            self.R_MAX = True
        elif self.Q_INIT_TYPE=='prior_knoledge':
            self.PRIOR_KNOWLEDGE = True
            self.X_SPLIT = 2
            self.Y_SPLIT = 3
            self.N_SUBAREAS = X_SPLIT*Y_SPLIT
        elif self.Q_INIT_TYPE=='zero':
            pass
        else:
            self.Q_INIT_TYPE = 'invalid'

        # [UAVs]



        self.section = config['UAVs']

        self.N_UAVS = int(self.section['N_UAVS'])
        self.UNLIMITED_BATTERY = True if self.section['BATTERY_AUTONOMY_TIME']=='None' else False
        if isinstance(self.section['BATTERY_AUTONOMY_TIME'], int):
            self.BATTERY_AUTONOMY_TIME = int(self.section['BATTERY_AUTONOMY_TIME'])
        else:
            if self.section['BATTERY_AUTONOMY_TIME']=='None':
                self.BATTERY_AUTONOMY_TIME = None
            else:
                try:
                    self.BATTERY_AUTONOMY_TIME = int(self.section['BATTERY_AUTONOMY_TIME'])
                except:
                    assert False, "Battery autonomy value not valid!"
        self.MIN_UAV_HEIGHT = int(self.section['MIN_UAV_Z'])
        self.NOISE_ON_POS_MEASURE = True if self.section['NOISE_ON_POS_MEASURE']=='True' else False
        self.MULTI_SERVICE = True if self.section['MULTI_SERVICE']=='True' else False
        self.N_SERVICES = 3 if self.MULTI_SERVICE==True else 1
        self.UAV_BANDWIDTH = float(self.section['UAV_BANDWIDTH'])
        self.UAV_FOOTPRINT = float(self.section['UAV_FOOTPRINT'])
        # The actual footprint is the footprint rescaled according to the selected resolution for columns (it will be a circular footprint even if resolution for columns and rows differ in such a way the footprint should be an ellipse):
        self.ACTUAL_UAV_FOOTPRINT = self.UAV_FOOTPRINT/self.CELL_RESOLUTION_PER_COL
        self.DELAYED_START_PER_UAV = int(self.section['DELAYED_START_PER_UAV'])
        # Number of traveled cells per action along XY axes;
        self.UAV_XY_STEP = 1
        # Number of traveled cells per action along Z axis;
        self.UAV_Z_STEP = 3
        # Agents actions:
        self.HOVERING = 4
        self.UP = 5
        self.DOWN = 6
        self.LEFT = 7
        self.RIGHT = 8
        self.RISE = 9
        self.DROP = 10
        self.GO_TO_CS = 1
        self.CHARGE = 2
        if self.HOSP_SCENARIO==False:
            self.UAVS_POS = []
            RAD_BETWEEN_POINTS = 2*pi/self.N_UAVS
            for uav_idx in range(self.N_UAVS):
                if (self.DIMENSION_2D == True):
                    self.UAVS_POS.append( (round(ceil(self.CELLS_COLS/2) + self.RADIAL_DISTANCE_X*sin(uav_idx*RAD_BETWEEN_POINTS)), round(ceil(self.CELLS_ROWS/2) + self.RADIAL_DISTANCE_Y*cos(uav_idx*RAD_BETWEEN_POINTS)), 0) )
                else:
                    self.UAVS_POS.append( (round(ceil(self.CELLS_COLS/2) + self.RADIAL_DISTANCE_X*sin(uav_idx*RAD_BETWEEN_POINTS)), round(ceil(self.CELLS_ROWS/2) + self.RADIAL_DISTANCE_Y*cos(uav_idx*RAD_BETWEEN_POINTS)), self.MIN_UAV_HEIGHT) )
        # Percentage of battery consumption per iteration:
        self.PERC_CONSUMPTION_PER_ITERATION = 4 #In this way it is possible to have integer battery levels without approximation errors which lead to 'Key Error' inside Q-table dictionary:
        self.FULL_BATTERY_LEVEL = self.ITERATIONS_PER_EPISODE*self.PERC_CONSUMPTION_PER_ITERATION # = iteration * 4
        PERC_BATTERY_CHARGED_PER_IT = 20
        self.BATTERY_CHARGED_PER_IT = round((PERC_BATTERY_CHARGED_PER_IT*self.FULL_BATTERY_LEVEL)/100)
        #PERC_CONSUMPTION_PER_ITERATION = STEP_REDUCTION_TO_GO_TO_CS
        self.PERC_BATTERY_TO_GO_TO_CS = int(self.PERC_CONSUMPTION_PER_ITERATION/4) # --> 4/4 = 1
        self.PERC_CRITICAL_BATTERY_LEVEL = 20
        self.CRITICAL_BATTERY_LEVEL = round((self.PERC_CRITICAL_BATTERY_LEVEL*self.FULL_BATTERY_LEVEL)/100)
        self.CRITICAL_BATTERY_LEVEL_2 = round(self.CRITICAL_BATTERY_LEVEL/1.5)
        self.CRITICAL_BATTERY_LEVEL_3 = round(self.CRITICAL_BATTERY_LEVEL_2/1.5)
        self.CRITICAL_BATTERY_LEVEL_4 = round(self.CRITICAL_BATTERY_LEVEL_3/1.5)
        self.CRITICAL_WAITING_TIME_FOR_SERVICE = 10
        self.NORMALIZATION_FACTOR_WAITING_TIME_FOR_SERVIE = 120
        self.SHOW_BATTERY_LEVEL_FOR_CHARGING_INSTANT = 400

        # [Users]

        self.section = config['Users']

        self.MAX_USER_Z = int(self.section['MAX_USER_Z'])
        self.INF_REQUEST = True if self.section['INF_REQUEST']=='True' else False
        self.STATIC_REQUEST = True if self.section['STATIC_REQUEST']=='True' else False
        self.USERS_PRIORITY = True if self.section['USERS_PRIORITY']=='True' else False
        self.MOVE_USERS_EACH_N_EPOCHS = int(self.section['MOVE_USERS_EACH_N_EPOCHS'])
        self.UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS = int(self.section['UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS'])
        self.CENTROIDS_MIN_MAX_COORDS = ast.literal_eval(self.section['CENTROIDS_MIN_MAX_COORDS'])
        self.FIXED_CLUSTERS_NUM = len(self.CENTROIDS_MIN_MAX_COORDS)
        # List of the number of clusters to test ONLY IF looking for the oprimal number of users clusters to detect:
        self.CLUSTERS_NUM_TO_TEST = [3, 4, 5, 6, 7, 8, 9, 10]

        # [Hospitals]

        #Config.hospitals_aux()

        from settings.hosp_scenario import hosp_features

        self.hospitals_ini = hospitals_ini
        hospitals_file = config.read(hospitals_ini)
        if hospitals_file==[]:
            print("ERROR cannot load settings init file")
            sys.exit(1)
        
        self.section = config['Hospitals']

        self.hosps_features = hosp_features()
        self.N_MISSION = int(self.section['N_MISSION'])
        self.UAV_HEIGHT_Z = int(self.section['UAV_HEIGHT_Z'])
        #self.N_MISSION = self.N_UAVS
        try:
            self.hosp_pos = ast.literal_eval(self.section['hosp_pos'])
        except:
            print("No valid hospitals locations!")
        #print(len(self.hosp_pos[0]), "weee")

        if (j != 1000):  # Aggiustare in un modo più elegante.. Se j = None, plot dello scenario finale.
            self.N_HOSP = len(self.hosp_pos[j]) if (isinstance(self.hosp_pos, list) and len(self.hosp_pos)>=1) else None
        else:
            self.N_HOSP = 2 #Numero fittizio

        count = 0
        for i in range(len(self.hosp_pos)):
            for k in range(len(self.hosp_pos[i])):
                count += 1
        self.N_HOSP_TOT = count  # N_HOSP_TOT è il numero di ospedali da plottare nella legenda (plotting.py)



        assert self.N_MISSION<=len(self.hosp_pos), "The number of missions must be equal or less than the number of hospitals"
        # Hospitals random locations generation:
        self.randomm = False if self.N_HOSP!=None else True 
        self.nb_colors = int(self.section['nb_colors'])
        self.HOSP_PRIORITIES = self.hosps_features.HOSP_PRIORITIES
        self.PRIORITY_NUM = self.hosps_features.PRIORITY_NUM
        if self.HOSP_SCENARIO==True:
            try:
                self.UAVS_POS = ast.literal_eval(self.section['UAVS_POS'])
            except:
                print("No valid UAVs locations for hospitals scenario")
                hosp_pos_ = []
                UAVS_POS_ = []
                # Load in scenario only hospitals and UAVs positions related to the selected number of missions:
                for idx in range(self.N_MISSION):
                    hosp_pos_.append(self.hosp_pos[idx])
                    UAVS_POS_.append(self.UAVS_POS[idx])
                self.hosp_pos = hosp_pos_
                self.UAVS_POS = UAVS_POS_


        # Action space
        self.ACTION_SPACE_3D_MIN = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.RISE, self.DROP, self.HOVERING] # TO DEFINE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.ACTION_SPACE_3D_COME_HOME = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.RISE, self.DROP, self.HOVERING, self.GO_TO_CS]
        self.ACTION_SPACE_3D_WHILE_CHARGING = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.RISE, self.DROP, self.HOVERING, self.CHARGE]
        self.ACTION_SPACE_3D_TOTAL = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.RISE, self.DROP, self.HOVERING, self.GO_TO_CS, self.CHARGE]
        self.GO_TO_CS_3D_INDEX = self.ACTION_SPACE_3D_TOTAL.index(self.GO_TO_CS)
        self.GO_TO_CS_3D_INDEX_HOME_SPACE = self.ACTION_SPACE_3D_COME_HOME.index(self.GO_TO_CS)
        self.CHARGE_3D_INDEX = self.ACTION_SPACE_3D_TOTAL.index(self.CHARGE)
        self.CHARGE_3D_INDEX_WHILE_CHARGING = self.ACTION_SPACE_3D_WHILE_CHARGING.index(self.CHARGE)

        self.ACTION_SPACE_2D_MIN = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.HOVERING]
        self.ACTION_SPACE_2D_COME_HOME = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.HOVERING, self.GO_TO_CS]
        self.ACTION_SPACE_2D_WHILE_CHARGING = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.HOVERING, self.CHARGE]
        self.ACTION_SPACE_2D_TOTAL = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.HOVERING, self.GO_TO_CS, self.CHARGE]
        self.GO_TO_CS_2D_INDEX = self.ACTION_SPACE_2D_TOTAL.index(self.GO_TO_CS)
        self.GO_TO_CS_2D_INDEX_HOME_SPACE = self.ACTION_SPACE_2D_COME_HOME.index(self.GO_TO_CS)
        self.CHARGE_2D_INDEX = self.ACTION_SPACE_2D_TOTAL.index(self.CHARGE)
        self.CHARGE_2D_INDEX_WHILE_CHARGING = self.ACTION_SPACE_2D_WHILE_CHARGING.index(self.CHARGE)

        if (self.DIMENSION_2D==True):
            if (self.UNLIMITED_BATTERY == True):  
                DICT_ACTION_SPACE_2D = {self.ACTION_SPACE_2D_MIN.index(self.UP): "UP", self.ACTION_SPACE_2D_MIN.index(self.DOWN): "DOWN", self.ACTION_SPACE_2D_MIN.index(self.LEFT): "LEFT", self.ACTION_SPACE_2D_MIN.index(self.RIGHT): "RIGHT", self.ACTION_SPACE_2D_MIN.index(self.HOVERING): "HOVER"}
            else:
                DICT_ACTION_SPACE_2D = {self.ACTION_SPACE_2D_TOTAL.index(self.UP): "UP", self.ACTION_SPACE_2D_TOTAL.index(self.UP): "DOWN", self.ACTION_SPACE_2D_TOTAL.index(self.UP): "LEFT", self.ACTION_SPACE_2D_TOTAL.index(self.UP): "RIGHT", self.ACTION_SPACE_2D_TOTAL.index(self.UP): "HOVER", self.ACTION_SPACE_2D_TOTAL.index(self.GO_TO_CS): "GO CS", self.ACTION_SPACE_2D_TOTAL.index(self.CHARGE): "CHARGE"}    
        else:
            if (self.UNLIMITED_BATTERY == True):
                DICT_ACTION_SPACE_3D = {self.ACTION_SPACE_3D_MIN.index(self.UP): "UP", self.ACTION_SPACE_3D_MIN.index(self.DOWN): "DOWN", self.ACTION_SPACE_3D_MIN.index(self.LEFT): "LEFT", self.ACTION_SPACE_3D_MIN.index(self.RIGHT): "RIGHT", self.ACTION_SPACE_3D_MIN.index(self.RISE): "RISE", self.ACTION_SPACE_3D_MIN.index(self.DROP): "DROP", self.ACTION_SPACE_3D_MIN.index(self.HOVERING): "HOVER"}
            else:
                DICT_ACTION_SPACE_3D = {self.ACTION_SPACE_3D_TOTAL.index(self.UP): "UP", self.ACTION_SPACE_3D_TOTAL.index(self.DOWN): "DOWN", self.ACTION_SPACE_3D_TOTAL.index(self.LEFT): "LEFT", self.ACTION_SPACE_3D_TOTAL.index(self.RIGHT): "RIGHT", self.ACTION_SPACE_3D_TOTAL.index(self.RISE): "RISE", self.ACTION_SPACE_3D_TOTAL.index(self.DROP): "DROP", self.ACTION_SPACE_3D_TOTAL.index(self.HOVERING): "HOVER", self.ACTION_SPACE_3D_TOTAL.index(self.GO_TO_CS): "GO CS", self.ACTION_SPACE_3D_TOTAL.index(self.CHARGE): "CHARGE"}

        self.DICT_ACTION_SPACE = DICT_ACTION_SPACE_2D if self.DIMENSION_2D==True else DICT_ACTION_SPACE_3D
        

        self.ACTION_SPACE_STANDARD_BEHAVIOUR = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.RISE, self.GO_TO_CS, self.CHARGE]
        self.LIMIT_VALUES_FOR_ACTION = [] # --> sarà una lista di tuple (min_val, max_val)

        self.MIDDLE_CELL_ASSUMPTION = True
        self.REDUCE_ITERATION_PER_EPISODE = False

        self.validate()

        #if self.HOSP_SCENARIO==True:
        #    self.N_UAVS = 1

    '''@staticmethod
    def hospitals_aux():
        with open("iter.py", "w") as text_file:
            text_file.write('j =' + '{}'.format(0))'''

    def validate(self):
        # Validating the values
        config = ConfigParser()

        # [Scenario]
        config.read(self.scenario_ini)
        self.section = config['Scenario']
        assert self.MIN_OBS_PER_AREA>=0 and self.MAX_OBS_PER_AREA<=1, "No valid coverage percentage for obstacle generation!"
        assert self.MIN_OBS_HEIGHT>=0, "No valid minimum height per building!"
        assert self.CELL_RESOLUTION_PER_COL>=1 and self.CELL_RESOLUTION_PER_ROW>=1, "No valid coverage percentage for cell resolution!"
        assert self.section['DIMENSION']=='2D' or self.section['DIMENSION']=='3D', "No valid scenario dimension!"
        assert self.section['SCENARIO_TYPE']=='hospitals' or self.section['SCENARIO_TYPE']=='user-service', "No valid type of scenario!"
        assert self.CS_HEIGHT<self.MAXIMUM_AREA_HEIGHT and self.CS_HEIGHT>=0, "No valid height for charging station!"
        assert (self.N_CS>=0 and self.UNLIMITED_BATTERY==False) or (self.N_CS<0 or self.N_CS>=0), "No valid number of charging stations!" 

        # [UAVs]
        assert self.N_UAVS>0 and self.N_UAVS<=self.ITERATIONS_PER_EPISODE, "No valid number of UAVs selected!"
        assert self.UNLIMITED_BATTERY!='invalid', "No valid battery autonomy value!"
        assert self.MIN_UAV_HEIGHT>=0, "No valid minimum flight height per UAV!"
        assert self.UAV_FOOTPRINT>=0, "No valid UAV footprint!"
        assert self.DELAYED_START_PER_UAV>=0 and (self.ITERATIONS_PER_EPISODE//self.DELAYED_START_PER_UAV>=1), "No valid iteration steps for delayed take-off!"

        # [Training]
        assert self.algorithm!='invalid', "No valid battery algorithm selection!"
        assert self.Q_INIT_TYPE!='invalid', "No valid Q-table initialization!"

        # [Users]

        if self.HOSP_SCENARIO==True:
            # [Hospitals]
            #assert self.nb_colors !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            assert self.N_MISSION > 0 and self.N_MISSION <= len(self.UAVS_POS) and self.N_MISSION <= len(self.hosp_pos), "Number of missions must be equal or less then UAVS_POS and hosp_pos"
            assert self.N_HOSP%2==0, "ODD HOSPITALS' NUMBER: you have to set an even number of hospitals in order to associate each of it with another one (i.e., each pair of hospitals will be made by a sender and a receiver)!"
            assert self.N_HOSP/self.PRIORITY_NUM%2 != 0, "PRIORITIES OR HOSPITALS NUMBER RESET NEEDED: you have to set a number of priorities which is the half of the number of hospitals/missions (indeed each mission is associated to a sender-receiver pair of hospitals)!"
            assert self.UNLIMITED_BATTERY==True, "No implemented case for hospitals and limited battery!"

    def inspect(self):
        print("============ ML config parameters ============")

        for k in self.__dict__.keys():
            print("%s: %r" %(k,self.__dict__[k]))

        print("-----------------------------------------------------------")

    def printkey(self,k):
        print("  %s = %r" %(k,self.__dict__[k]))

    def summary(self):
        # [Scenario]
        print("============ Scenario parameters ============")

        self.printkey('HOSP_SCENARIO')
        self.printkey('AREA_WIDTH')
        self.printkey('AREA_HEIGHT')
        self.printkey('MAXIMUM_AREA_HEIGHT')
        self.printkey('CELL_RESOLUTION_PER_COL')
        self.printkey('CELL_RESOLUTION_PER_COL')
        self.printkey('CREATE_ENODEB')
        self.printkey('DIMENSION_2D')
        self.printkey('CS_HEIGHT')
        self.printkey('MIN_OBS_PER_AREA')
        self.printkey('MAX_OBS_PER_AREA')
        self.printkey('MIN_OBS_HEIGHT')
        self.printkey('MAX_OBS_HEIGHT')

        # [UAVs]
        print("============ UAVs parameters ============")

        self.printkey('UNLIMITED_BATTERY')
        self.printkey('BATTERY_AUTONOMY_TIME')
        self.printkey('MIN_UAV_HEIGHT')
        self.printkey('NOISE_ON_POS_MEASURE')
        self.printkey('MULTI_SERVICE')
        self.printkey('UAV_BANDWIDTH')
        self.printkey('UAV_FOOTPRINT')
        self.printkey('DELAYED_START_PER_UAV')

        # [Training]
        print("============ Training parameters ============")

        self.printkey('algorithm')
        self.printkey('EPISODES')
        self.printkey('ITERATIONS_PER_EPISODE')
        self.printkey('REDUCE_ITERATION_PER_EPISODE')
        self.printkey('NO_INTERFERENCE_REWARD')
        self.printkey('Q_INIT_TYPE')

        # [Users]
        if self.HOSP_SCENARIO==False:
            print("============ Users parameters ============")

            self.printkey('MAX_USER_Z')
            self.printkey('INF_REQUEST')
            self.printkey('STATIC_REQUEST')
            self.printkey('USERS_PRIORITY')
            self.printkey('MOVE_USERS_EACH_N_EPOCHS')
            self.printkey('UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS')

        return ""