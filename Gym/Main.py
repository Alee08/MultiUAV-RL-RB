import os
import csv
from subprocess import Popen

with open("iter.py", "w") as text_file:
    text_file.write('j =' + '{}'.format(0))

myfile_traj = "./Best_policy/trajectory.csv"
myfile_plot = "./Best_policy/policy_per_plot_.csv"
## If file exists, delete it ##
if os.path.isfile(myfile_traj):
    os.remove(myfile_traj) #Elimino il csv delle traiettorie salvate in precedenza'''
else:
    pass
if os.path.isfile(myfile_plot):
    os.remove(myfile_plot) #Elimino il csv delle traiettorie salvate in precedenza'''
else:
    pass

policy_n = []
j = 0
N_MISSION = 10



for j in range(N_MISSION):


    with open("iter.py", "w") as text_file:
        text_file.write('j =' + '{}'.format(j))

    os.system('python3 scenario_objects.py')
    os.system('python3 plotting.py')
    os.system('python3 env_wrapper.py')
    j = j + 1

os.system('python3 scenario_objects_mission.py')
os.system('python3 plotting_mission.py')


