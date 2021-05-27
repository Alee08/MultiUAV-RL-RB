# UAV-RL-RB



## Docker
Install Docker:
https://docs.docker.com/get-docker/

### Using Jupyter
Build Docker Ubuntu based image named docker-uav-rl-rb from Dockerfile:
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