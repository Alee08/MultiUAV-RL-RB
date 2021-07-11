# MultiUAV-RL-RB
### Table of Contents

- [Description](#description)
- [Install & Run](#install-&-Run)
- [Outputs](#outputs)
- [Scenarios Setup](#scenario-setup)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Description
MultiUAV-RL-RB is a framework allowing you to generate 2 main difference scenario by setting different parameter for both of them.
In both scenarios the environment involves a volume (or area) of operations which is partitioned into K small cells whose size A<sub>k</sub>, k=0 . . . K-1.

### Service Scenario
In this scenario N drones can be deployed and M Charging Stations (CS) are made available at locations placed in points which are equidistant from the middle of the operational volume (area). These charging stations are placed at equal distance one from another. Drones are supposed to serve users while flying (except for some particular cases such as while they are going to a CS) and their performance are evaluated according to some predefined metrics. Each user requests one out of N_serv services (as throughput request, edge-computing or data gathering), with some parameters stored as attributes of each user.

Obstacles can be randomly generated by setting the coverage percentage of the obstacles for the whole operational volume (area). Maximum building height can be selected and, according to this choice, different heights for each building are randomly generated in a range going from the floor to the maximum selected value.

We can set the presence of C clusters of users, whose centroid locations are generated by drawing samples from a uniform distribution; we can also set U users, spread out along the generated centroids by using a truncated (bounded) normal distribution.

### Hospitals Scenario

MultiUAV-RL-RB is a fremework developed for the final thesis project in Engineering in Computer Science at Sapienza University of Rome, supervised by Prof. Luca Iocchi. The name of the master's thesis is: [Multi-UAV reinforcement learning with temporal and priority goals](https://alee08.github.io/Multi-UAV_RL_RB).

The goal of MultiUAV-RL-RB is to find an optimal policy for each agent, reducing interference between agent policies. Agents must respect the mission priority associated with each goal and reach the goals in the correct order.

<table>
  <thead>
    <tr>
      <th style="text-align: center"><img src="https://alee08.github.io/_pages/Multi_UAV/ex3.gif" alt="" /></th>
      <th style="text-align: center"><img src="https://alee08.github.io/_pages/Multi_UAV/ex4.gif" alt="" /></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">Experiment 3. Without priority</td>
      <td style="text-align: center">Experiment 4. With priority</td>
    </tr>
  </tbody>
</table>

## Scenarios Setup

You can setup different scenarios and different environment/training parameters by setting different values for each relevant parameter inside ***.ini*** files (inside ***settings*** folder). .ini files are the folowing one (each one named with a meaningful name):

1. scenario.ini
2. UAVs.ini
3. hospitals.ini
4. users.ini
5. training.ini

### scenario.ini
Here you can set all the parameters related to the environment (e.g., dimension, type of scenario, ...):
```
# Choose between 'hospitals' and 'multi-service':
SCENARIO_TYPE = hospitals
# Operating volume delimitations in cells (always starting from 0):
X = 10
Y = 10
Z = 12
# Select the cell resolution (indicates how many sub-cells will form a single cell) for X,Y (Z axis resoution is always set to 1):
X_CELL_RESOLUTION = 1
Y_CELL_RESOLUTION = 1
# Height (in cells) for each charging station (a good choice could be to set it equal to 'MIN_UAV_Z'):
CS_HEIGHT = 4
# Enable/Disable 2D environment (select between '2D' and 3D')
DIMENSION = 2D
# Obstacle coverage percentage w.r.t. to the operating selected area/volume:
MIN_OBS_PER_AREA = 0.05
MAX_OBS_PER_AREA = 0.15
MIN_OBS_HEIGHT = 3
# A good choice for the maximum height of an obstacle (e.g., building) could be to set it equal to 'Z_MAX':
MAX_OBS_HEIGHT = 12
# Number of charging stations (used only if the UAVs battery is not unlimited):
N_CS = 2
# Create EnodeB in the middle of the generated map (not fully implemented yet, thus it will be not considered during the training phase):
CREATE_ENODEB = False
```

### UAVs.ini

This file allows you to set all the parameters related to the UAVs (e.g., number of UAVs, battery duration, ...):
```
[UAVs]

# Number of UAVs:
N_UAVS = 1
# UAVs battery autonomy battery (in seconds): e.g., 1800s=30m. If set to None, then battery will be assumed to be unlimited:
BATTERY_AUTONOMY_TIME = None
# Minimum flight altitude (in cells) for each UAV:
MIN_UAV_Z = 4
# Enable/Disable random error on the position of each UAV used:
NOISE_ON_POS_MEASURE = False
# Enable/Disable multi-service (if disabled, only a single service task will be performed):
MULTI_SERVICE = False
# Enable/Disable limitation on UAV bandwidth (KHz ~ [Kbps])
UAV_BANDWIDTH = 5000
# Set the UAV footprint (radius) to make UAVs able to serve a smaller or bigger range of users:
UAV_FOOTPRINT = 2.5
# Iteration steps between an UAV take-off and the next one (it must be selected such that al the UAVs will take-off in the iterations of the first epoch):
DELAYED_START_PER_UAV = 6 
```

### users.ini

Here you can set all the users features (e.g., motion, users requests, ...) in case you are using a ***service scenario***:

```
[Users]

# Coordinates ranges [[(minX,maxX), (minY,maxY)], ...] for random (inside the selected ranges) centroids users clusters generation: 
CENTROIDS_MIN_MAX_COORDS = [[(2,2), (5, 5)]]
# Maximum height achievable by users (e.g., if they are in a building or balcony, ...)
MAX_USER_Z = 3
# Enable/Disable continuous users requests
INF_REQUEST = True
# Enable/Disable static (i.e., not moving users) request
STATIC_REQUEST = True 
# Not implemented yet (if enabled, it will not have any effect on the training phase):
USERS_PRIORITY = False
# How much often users are expected to randomly move on the selected map:
MOVE_USERS_EACH_N_EPOCHS = 300
# Ho much often (in iteration steps) the users are expected to perform a new service request:
UPDATE_USERS_REQUESTS_EACH_N_ITERATIONS = 10
```

## training.ini

This file allows you to set all the parameters related to the training phase (e.g., number of epochs, Q-table initialization method, ...):

```
[Training]

# Select the algorithm to use between 'baseline', 'q_learning', 'sarsa' and 'sarsa_lambda':
ALGORITHM = q_learning
# Number of epochs: 
EPISODES = 10000
# If True, it reduces the next episode iterations based on the best number of iterations per episode found until the current episode:
REDUCE_ITERATION_PER_EPISODE = False
# Number of iterations per epoch:
ITERATIONS_PER_EPISODE = 80
# If True, it takes into account the other trajectories agents:
NO_INTERFERENCE_REWARD = False
# Q-Table initialization (choose between 'zero', max_reward' and 'prior_knowledge'):
Q_INIT_TYPE = zero
```

### Settings

- Start position for each agent
- Priority goal for each agent
- LTLf / LDLf formula for each agent
- Fluents for each mission
- Position for each hospital
- Temporal goal for each hospital
- Training with agents considering priority (`NO_INTERFERENCE_REWARD` is `True`) or not (`NO_INTERFERENCE_REWARD` is `False`) 

### Dependencies

#### Programming Language

- Python (version >= 3.6)

#### Dependencies 

- gym (0.10.5)
- Numpy (1.15.2)
- scipy
- sklearn (0.19.1)
- Matplotlib (2.1.2)
- mpl_toolkits (the same as Matplotlib)
- ImageMagick
- pythomata (0.2.0)
- graphviz (0.16)

---

## Install & Run
It is possible to install the framework in two different ways, via [Docker](Docker) or via [Pip](Pip)
### Docker
Install Docker:
https://docs.docker.com/get-docker/

#### Using Jupyter
Build Docker Ubuntu based image named `docker-uav-rl-rb` from Dockerfile:
```console
$ docker build -t docker-uav-rl-rb:latest .
```
Run image opening port 8888 (change it if already in use):
```console
$ docker run -p 8888:8888 docker-uav-rl-rb:latest 
```
Now that our image is running...
1. Open the link specified in terminal
2. In browser you should see the current file structure
3. Open the notebook file main.ipynb 
4.  **run all cells**

### Pip
Navigate through your terminal/prompt to the folder containing `setup.py`, i.e. to `MultiUAV_RL_RB/Gym/`, and type:
```console
$ pip install -e .
```
Now all needed dependencies are installed.

#### Run

Now run:
```console
$ Python3 Main.py 
```

---

## Outputs

Inside the `Best Policy/` directory we get four csv files. 

The `Flights_trajectories/` directory contains the `MissionStart_%Y-%m-%d %H:%M:%S/` directory renamed with the mission start time and date.
Inside each `MissionStart_%Y-%m-%d %H:%M:%S/` folder there is the best policy in csv format, renamed with the identification ID of the UAV. 

The directory structure of `Gym/` for reading outputs is:
```
Gym
├── Best_policy/
│   ├── best_policy_obs.csv                   # CSV file with the observations of the best_policy
│   ├── Best_policys.csv                      # CSV file in format: 'UAS id', 'UAS Relative time', 'x', 'y', 'z' 
│   ├── policy_per_plot_.csv                  # CSV used for the plot of the experiments
│   └── trajectory.csv                        # CSV containing all the x, y cells that the UAVs passed through 
│
├── Flights_trajectories/
│   ├── MissionStart_'%Y-%m-%d %H:%M:%S'/     # Directory renamed with mission start date and time
│   │   └── UAV-0_BestPolicy.csv              # CSV-ID file in format: 'UAS id', 'UAS Relative time', 'x', 'y', 'z'
│   ├── MissionStart_'%Y-%m-%d %H:%M:%S'/     # see above
│   │   └──  UAV-1_BestPolicy.csv             # see above
│   ...                                       # see above
│       └── UAV-99_BestPolicy.csv/            # see above
│
└── Cases/                                    #
    ...                                       # 
```

### Best_policy.csv
The `BestPolicy.csv` contains the mission of each UAV. 

Each UAV has an `ID`. The UAV with the lowest ID has the highest priority. 
`T` stands for time and indicates at which iteration the UAV is in the cell with coordinates `x`, `y` and `z`. 

The `Z-axis` in the example is set to `0` because it is not considered.


|ID  |T  |x  |y  |z  |
|---|---|---|---|---|
|0  |0  |0  |5  |0  |
|0  |1  |1  |5  |0  |
|0  |2  |1  |6  |0  |
|...  |...  |...  |...  |...  |
|1  |0  |1  |8  |0  |
|1  |1  |2  |8  |0  |
|1  |2  |3  |8  |0  |
|...  |...  |...  |...  |...  |
|2  |0  |10 |3  |0  |
|2  |1  |10 |4  |0  |
|2  |2  |10 |5  |0  |
|...  |...  |...  |...  |...  |

---

## References

...

---

## License

This project is intended for private use only. All rights are reserved and it is not Open Source or Free. You cannot modify or redistribute this code without explicit permission from the copyright holder. 

---

## Author Info

- Name: Alessandro
- Surname: Trapasso
- University: Sapienza University of Rome
- Master degree: [Engieering in Computer Science](https://corsidilaurea.uniroma1.it/it/corso/2020/30430/home)
- University e-mail: trapasso.1597997@studenti.uniroma1.it
- Private e-mail ale.trapasso8@gmail.com 

[Back To The Top](#read-me-template)
