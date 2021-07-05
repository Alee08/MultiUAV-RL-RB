import os
from subprocess import Popen

with open("iter.txt", "w") as text_file:
    text_file.write('{}'.format(0))

from configuration import Config
import shutil

conf = Config()


myfile_initial_users = "./initial_users/"
myfile_map_data = "./map_data/"
myfile_map_status = "./map_status/"
myfile__pycache__ = "./__pycache__/"

shutil.rmtree(myfile_initial_users, ignore_errors=True)
shutil.rmtree(myfile_map_data, ignore_errors=True)
shutil.rmtree(myfile_map_status, ignore_errors=True)
shutil.rmtree(myfile__pycache__, ignore_errors=True)


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
#N_MISSION = 3



for j in range(conf.N_MISSION):


    with open("iter.txt", "w") as text_file:
        text_file.write('{}'.format(j))


    os.system('python3 scenario_objects.py')
    os.system('python3 plotting.py')
    os.system('python3 env_wrapper.py')
    if conf.N_MISSION > 1:
        j = j + 1


with open("iter.txt", "w") as text_file:
    text_file.write('{}'.format(1000)) #Il numero 1000 indica lo scenario finale.

if conf.HOSP_SCENARIO == True:
    os.system('python3 scenario_objects.py')
    os.system('python3 plotting.py')


