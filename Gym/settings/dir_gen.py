import csv
import os
from os import mkdir
from os.path import isdir
from datetime import datetime
import ast
from configuration import Config

conf = Config()

traj_j = []
policy_per_plot = []
plot_policy = []
traj_j_ID = []

CSV_DIRECTORY_NAME = "Flights_trajectories"
BP_DIRECTORY_NAME = 'Best_policy/'
myfile = "./Best_policy/trajectory.csv"
myfile_plot = "./Best_policy/policy_per_plot_.csv"


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