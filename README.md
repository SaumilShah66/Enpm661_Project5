# Astar_Search_Algorithm_ENPM661-Project-3 Phase4

## Overview

Project 5 for course work ENPM661 - "Planning for autonomous robots". This project was completed with the collaborative efforts of 
-- Varun Asthana
-- Saumil Shah
-- Smriti Gupta

### Dependencies
* ROS Kinetic
* Gazebo7
* python2.7
* numpy
* matplotlib
* openCV (version 3.3)
* scipy

## Installing Turtlebot package for simulation

```
$ sudo apt-get install ros-kinetic-turtlebot-gazebo ros-kinetic-turtlebot-apps ros-kinetic-turtlebot-rviz-launchers
```

### How to build and run the program
```
$ mkdir planner_ws
$ cd planner_ws && cd src
$ git clone https://github.com/SaumilShah66/Enpm661_Project5
```

Ensure the executable access is give to the 'botmove.py' and 'astar.py' and also do not clone the entire repository inside the catkin/src directory (it affect the catkin_make command)

```
$ cd catkin_ws
$ catkin_make
$ source devel/setup.bash
$ roslaunch turtlebot_path_planner_astar planner.launch c:=0.02 x:=-2 y:=2 t:=0
```

In the last command above, parameter c is for robot clearance, (x,y,t) are for initial pose as coordinates and theta with origin at the center of the map and theta is measured anti-clockwise from positive x-axis. All parameters are to be in METERS except for 't' to be in degrees.

__Note: No control law has been implemented to check that the same trajectory is followed in simulation. Hence It is recommended to use near by start and goal points and also avoid any start point which would require a turn of 180 degrees (initially) to reach the goal. Also because of the same reason, it is recommended to use a clearance of 0.2 or more.__

### User Inputs after launching of ros nodes
All inputs are to be given in METERS
* Left and right wheel RPM (eg: 90,90)
* Goal position in x,y (eg: 0,-3)

RPM is converted to the unit of "rad/sec" by multiplying the user input with 2pi/60.

Also, ceil value of radius + clearance will be considered. Threshold for reaching the goal is set at 0.1m (or 10cm). 

### Path Generation
The run speed from top left to bottom right of the map (withut plotting of explored nodes) is around 5 mins.

After the goal point is reached, a path will be traced back from the start to goal point. On the map, this path will be drawn in RED. In the file location, an image "back_tracking.png" is saved. Once the path is displayed, the user can type any number and press enter to start the simulation in Gazebo.
