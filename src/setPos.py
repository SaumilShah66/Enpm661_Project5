#!/usr/bin/env python

import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
import argparse
import numpy as np
from Phase3 import *

def setInitialPos(start):
    print "Trying to set it to initial position"
    rospy.sleep(5.0)
    x = start[0]
    y = start[1]
    a = start[2]*3.14/180.0
    # print(x,y,a)
    state_msg = ModelState()
    state_msg.model_name = 'mobile_base'
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = 0
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = np.sin(a/2)  
    state_msg.pose.orientation.w = np.cos(a/2)
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )

    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

    print "Done....Settled to initial position"
# def get

def moverobot(args, solver, velocities=None):
    r = float(args.WheelRadius)
    l = float(args.WheelLength)

    pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
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
            # rospy.sleep(0.1)
        # cmd.linear.x = 0 
        # cmd.linear.y = 0
        # pub.publish(cmd)
        
def main():
    rospy.init_node('set_pose')
    Parser = argparse.ArgumentParser()
    # tuck_group = parser.add_mutually_exclusive_group(required=True)
    Parser.add_argument("--Start", default="[-4.5,4.0,120]", help="Initial X position")
    Parser.add_argument('--End', default="[0, -3, 0]", help='Give final point')
    Parser.add_argument('--RobotRadius', default=0.177, help='Give robot radius')
    Parser.add_argument('--Clearance', default=0.05, help='Give robot clearance')
    Parser.add_argument('--ShowExploration', default=0, help='1 if want to exploration animation else 0')
    Parser.add_argument('--ShowPath', default=1, help='1 if want to show explored path else 0')
    Parser.add_argument('--thetaStep', default=30, help='Possibilities of action for angle')
    Parser.add_argument('--StepSize', default=2, help='Step size')
    Parser.add_argument('--Threshold', default=0.01, help='Threshold value for appriximation')
    Parser.add_argument('--GoalThreshold', default=0.1, help='Circle radius for goal point')
    Parser.add_argument('--WheelRadius', default=0.038, help='Radius of the robot wheel in meters')
    Parser.add_argument('--WheelLength', default=0.320, help='Distance between two wheels')
    Parser.add_argument('--RPM', default="[10,10]", help='RPM of left wheel')
    Parser.add_argument('--Weight', default=1, help='Weight for cost to go')

    Args = Parser.parse_args(rospy.myargv()[1:])
    initial = [float(i) for i in Args.Start[1:-1].split(',')]
    end = Args.End
    goal = [float(i) for i in end[1:-1].split(',')] 
    print(goal)
    print(initial)
    r = float(Args.RobotRadius)
    c = float(Args.Clearance)
    
    StepSize = int(Args.StepSize)
    Threshold = float(Args.Threshold)
    GoalThreshold = float(Args.GoalThreshold)
    wheelLength = float(Args.WheelLength) 
    rpm = [float(i) for i in Args.RPM[1:-1].split(',')]
    Ur, Ul = rpm[0], rpm[1]
    print(Ur, Ul) 
    wheelRadius = float(Args.WheelRadius)
    weight = float(Args.Weight)

    rospy.loginfo("Setting initial position")
    setInitialPos(initial)
    rospy.loginfo("Settled to initial position")

    solver = pathFinder(initial, goal, stepSize=StepSize,
        goalThreshold = GoalThreshold, width = 10, height = 10, threshold = Threshold,
        r=r, c=c, wheelLength = wheelLength, Ur = Ur, Ul = Ul, wheelRadius = wheelRadius,
        weight = weight, showExploration=int(Args.ShowExploration), showPath=int(Args.ShowPath))
    rospy.loginfo("Trying to find the path  "+"--"*50)
    solver.findPath()
    rospy.loginfo("Done "+"="*50)
    moverobot(Args, solver)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass