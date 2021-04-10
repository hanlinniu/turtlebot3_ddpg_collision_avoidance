# Project Title: Mapless Collision Avoidance of Turtlebot3 Mobile Robot Using DDPG and Prioritized Experience Replay
This work is implemented in paper [Accelerated Sim-to-Real Deep Reinforcement Learning: Learning Collision Avoidance from Human Player](https://arxiv.org/abs/2102.10711) published in 2021 IEEE/SICE International Symposium on System Integration (SII) and [Voronoi-Based Multi-Robot Autonomous Exploration in Unknown Environments via Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9244647) published in IEEE Transactions on Vehicular Technology.

Demo Video 1: [Link](https://youtu.be/BmwxevgsdGc) 

Demo Video 2: [Link](https://youtu.be/XYvwVYhxP-o) 

Single Turtlebot3 Collision Avoidance:

![](single-turtlebot-collision-avoidance.gif)

Multiple Turtlebot3 Collision Avoidance:

![](multiple-turtlebot-collision-avoidance.gif)

This code is for training and testing ddpg algorithm on turtlebot3 waffle pi.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software

```
Ubuntu 16.04
ROS Kinetic
Tensorflow-gpu == 1.13.1 or 1.14.0
Keras == 2.3.1
```


### Virtual Environment

You need to make a virtual environment called 'ddpg_env' and install the following library for it.

```
Tensorflow-gpu == 1.13.1 or 1.14.0
Keras == 2.3.1
```

### Installing

The next step is to install dependent packages for TurtleBot3 control on Remote PC. For more details, please refer to [turtlebot3](http://emanual.robotis.com/docs/en/platform/turtlebot3/setup/#setup).

```
$ sudo apt-get update
$ sudo apt-get upgrade
$ wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_kinetic.sh && chmod 755 ./install_ros_kinetic.sh && bash ./install_ros_kinetic.sh

$ sudo apt-get install ros-kinetic-joy ros-kinetic-teleop-twist-joy ros-kinetic-teleop-twist-keyboard ros-kinetic-laser-proc ros-kinetic-rgbd-launch ros-kinetic-depthimage-to-laserscan ros-kinetic-rosserial-arduino ros-kinetic-rosserial-python ros-kinetic-rosserial-server ros-kinetic-rosserial-client ros-kinetic-rosserial-msgs ros-kinetic-amcl ros-kinetic-map-server ros-kinetic-move-base ros-kinetic-urdf ros-kinetic-xacro ros-kinetic-compressed-image-transport ros-kinetic-rqt-image-view ros-kinetic-gmapping ros-kinetic-navigation ros-kinetic-interactive-markers

$ cd ~/catkin_ws/src/
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
$ cd ~/catkin_ws && catkin_make
```


## Setting up the network between PC and turtlebot

Please refer to [Turtlebot3 Setup](http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#install-ubuntu-on-remote-pc)

## Git clone ddpg scripts

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance.git
$ cd ~/catkin_ws && catkin_make
```

### Start gazebo world (Change your world file location based on your setting)
To launch the corridor world
```
$ roslaunch turtlebot_ddpg turtlebot3_empty_world.launch world_file:='/home/hanlin/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/worlds/turtlebot3_modified_corridor2.world'
```
To launch the maze world
```
$ roslaunch turtlebot_ddpg turtlebot3_empty_world.launch world_file:='/home/hanlin/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/worlds/turtlebot3_modified_maze.world'
```

### Start a new terminal and Open python virtual environment

```
$ source ~/ddpg_env/bin/activate
```

For train and play with original ddpg

```
$ cd ~/catkin_ws/src/UGV_CA_ddpg/turtlebot_ddpg/scripts/original_ddpg
$ rosrun turtlebot_ddpg ddpg_network_turtlebot3_original_ddpg.py
```


For train and play with ddpg with human data
```
$ cd ~/catkin_ws/src/UGV_CA_ddpg/turtlebot_ddpg/scripts/fd_replay/play_human_data
$ rosrun turtlebot_ddpg ddpg_network_turtlebot3_amcl_fd_replay_human.py
```

For training
```
please change train_indicator=1 under ddpg_network_turtlebot3_original_ddpg.py or ddpg_network_turtlebot3_amcl_fd_replay_human.py
```

For playing trained weights
```
 please change train_indicator=0 under ddpg_network_turtlebot3_original_ddpg.py or ddpg_network_turtlebot3_amcl_fd_replay_human.py
```



## Authors

* **Hanlin Niu** - [Personal Page](https://www.research.manchester.ac.uk/portal/hanlin.niu.html)


### Paper
If you use this code in your research, please cite our IEEE Transactions on Vehicular Technology paper:
```
@article{hu2020voronoi,
  title={Voronoi-based multi-robot autonomous exploration in unknown environments via deep reinforcement learning},
  author={Hu, Junyan and Niu, Hanlin and Carrasco, Joaquin and Lennox, Barry and Arvin, Farshad},
  journal={IEEE Transactions on Vehicular Technology},
  volume={69},
  number={12},
  pages={14413--14423},
  year={2020},
  publisher={IEEE}
}
```
