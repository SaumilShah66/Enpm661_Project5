#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from astar import Astar
import time

def moverobot(args, solver, velocities=None):
    r = float(args.WheelRadius)
    l = float(args.WheelLength)

    pub = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
    cmd = Twist()
    
    rate = rospy.Rate(10.0)
    rospy.loginfo("Waiting")
    rospy.sleep(10.0)
    rospy.loginfo("Moveing now")
    rospy.loginfo(len(solver.trackIndex))

    for idx in solver.trackIndex:
        ul, ur = solver.actions[int(idx)] 
        vx = r*0.5*(ur+ul)
        rz = r*(ur-ul)/l
        start = time.time()
        while(time.time() - start <= 1):
            cmd.linear.x = vx 
            cmd.angular.z = rz
            pub.publish(cmd)

class Node(object):
    def __init__(self):
        # Params
        self.data = Twist()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # Publishers
        self.pub = rospy.Publisher("/cmd_vel_mux/input/navi", Twist, queue_size=10000)
        self.loop_rate.sleep()

    def pubData(self, lin, rot):
        start = time.time()
        while(time.time() - start <= 1):
            self.data.angular.z= rot
            self.data.linear.x = 0
            self.pub.publish(self.data)
            self.loop_rate.sleep()
        start = time.time()
        while(time.time() - start <= 0.75):
            self.data.angular.z= 0
            self.data.linear.x = lin
            self.pub.publish(self.data)
            self.loop_rate.sleep()
        pass

if __name__ == '__main__':
    rospy.init_node("botmove", anonymous=False)
    init_x = rospy.get_param('~init_x')
    init_y = rospy.get_param('~init_y')
    init_theta = rospy.get_param('~init_t')
    clrn = [float(rospy.get_param('~clr'))]
    
    my_node = Node()
    initCord = [init_x, init_y, init_theta]

    actionSet, step = Astar(initCord, clrn)
    print("\nactions--\n\n",actionSet,"\nstep\n",step)
    time.sleep(2)
    if(len(actionSet) != 0):
        for angle in actionSet[1:]:
            x = step*1.0/50
            if angle>180:
                angle = angle-360
            z = angle*3.14/180

            my_node.pubData(0.7*x, z)
            # my_node.pubData(0.812*x,0.965*z)
