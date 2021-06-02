# MultiUAV-RL-RB
### Table of Contents

- [Description](#description)
- [Install & Run](#install-&-Run)
- [Outputs](#outputs)  
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Description
MultiUAV-RL-RB is a fremework developed for the final thesis project in Engineering in Computer Science at Sapienza University of Rome, supervised by Prof. Luca Iocchi. The name of the master's thesis is: [Multi-UAV reinforcement learning with temporal and priority goals](https://alee08.github.io/Multi-UAV_RL_RB).

The goal of MultiUAV-RL-RB is to find an optimal policy for each agent, reducing interference between agent policies. Agents must respect the mission priority associated with each goal and reach the goals in the correct order.
###Hosp Scenario
<div>
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
</div>

###Settings

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

####Run
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