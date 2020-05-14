#!/usr/bin/env python
import roslib
roslib.load_manifest('usi_angry_turtle')
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import time
import math
from std_srvs.srv import Empty as EmptyServiceCall
from std_msgs.msg import String
from math import pow, atan2, sqrt
import turtlesim.srv
import random
import std_srvs.srv
from turtlesim.srv import SetPen
PI = 3.1415926535897
pi = 3.1415926535897
import tf
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
from turtlesim.srv import Spawn

x=0.0
y=0.0
z=0.0
lastDistance = 10
yaw=0.0
turtle2x = 0.0
turtle2y = 0.0
distance = 0.0
ACDC = False

velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
tolerance = 0.0

def poseCallback(pose_message):
    global x
    global y, z, yaw
    x= round(pose_message.x,4)
    y= round(pose_message.y,4)
    yaw = round(pose_message.theta,4)


def turtle2Pose(pose_message):
    global turtle2x, turtle2y, turtle2yaw, distance, tolerance, ACDC
    turtle2x =round(pose_message.x,4)
    turtle2y = round(pose_message.y,4)
    turtle2yaw =round(pose_message.theta,4)
    distance = sqrt(pow((turtle2x-x),2) + pow((turtle2y-y),2))
    if (distance > tolerance):
        ACDC = False
        print("I am an annoying beach")
    else: 
        ACDC = True
        print("Wait for me beach")
    


def move(speed, distance):
    global velocity_publisher
        #declare a Twist message to send velocity commands
    vel_msg_m = Twist()
    #get current location 
    x0=x
    y0=y
    #z0=z;
    #yaw0=yaw;
    vel_msg_m.linear.x =speed
    distance_moved = 0.0
    loop_rate = rospy.Rate(10) # we publish the velocity at 10 Hz (10 times a second)    
    cmd_vel_topic='/turtle1/cmd_vel'
    velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

    while True :
            rospy.loginfo("Turtlesim moves forwards")
            velocity_publisher.publish(vel_msg_m)

            loop_rate.sleep()
            
            #rospy.Duration(1.0)
            
            distance_moved = distance_moved+abs(0.5 * math.sqrt(((x-x0) ** 2) + ((y-y0) ** 2)))
            print  distance_moved               
            if  not (distance_moved<distance):
                rospy.loginfo("reached")
                break
    
    #finally, stop the robot when the distance is moved
    vel_msg_m.linear.x =0
    velocity_publisher.publish(vel_msg_m)


def rotate(speed, angle, clockwise):
    
    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    # Receiveing the user's input
    print("Let's rotate your robot")
    #Converting from angles to radians
    angular_speed = speed*2*PI/360
    relative_angle = angle*2*PI/360

    #We wont use linear components
    vel_msg.linear.x=0
    vel_msg.linear.y=0
    vel_msg.linear.z=0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    # Checking if our movement is CW or CCW
    if clockwise:
        vel_msg.angular.z = -abs(angular_speed)
    else:
        vel_msg.angular.z = abs(angular_speed)
    # Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    current_angle = 0

    while(current_angle < relative_angle):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed*(t1-t0)


    #Forcing our robot to stop
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)

def getDistance():   
    global turtle2x, turtle2y, x, y, yaw   
    distance = sqrt(pow((turtle2x-x),2) + pow((turtle2y-y),2))
    print("distance "+ str(distance))
    return distance

def tutorialMethodPlus():
    global lastDistance
    motion = Twist()
    distance = getDistance()
    targetTheta = atan2(turtle2y - y, turtle2x - x)
    motion.linear.x = 1.5*distance
    change = 1
    if (targetTheta - yaw <0):
        change = -1
    motion.angular.z = 4 * (targetTheta - yaw)* change 
    velocity_publisher.publish(motion)  
    loop_rate.sleep()
    # if (targetTheta<0):
    #    targetTheta += 2 * pi
    # if (distance <= lastDistance):
    #     motion.linear.x = 1.5*distance
    # else:
    #     motion.linear.x = .1
    # lastDistance=distance
    # change = 1
    # if (targetTheta - yaw <0):
    #     change = -1
    # motion.angular.z = 4 * (targetTheta - yaw) * change
    # velocity_publisher.publish(motion)  
    # loop_rate.sleep()
    
    ##print(str(turtle2x))
    #print(str(x))
    #print(" y "+ str(turtle2y))

    #print(str(y))

    #print(" tartget theta "+ str(targetTheta))

def usi_marker():
     #first declare at init the instantiated variable
    global ACDC
    pen = SetPen()
    pen = rospy.ServiceProxy('/turtle1/set_pen', SetPen)

    pen(255,255,255,1,0)

    rotate(30 , 97, 0)  # up            

    time.sleep(2)
    print 'move: '
    move (1.0, 17.0)
    time.sleep(2)
    if (ACDC == True):
        return

    rotate(30 , 180, 1) 

    time.sleep(2)
    print 'move: '
    move (1.0, 17.0)
    time.sleep(2)
    if (ACDC == True):
        return

    rotate(30 , 90, 1)

    time.sleep(2)
    print 'move: '
    move (1.0, 5.0)
    time.sleep(2)
    if (ACDC == True):
        return

    rotate(30 , 90, 1)

    time.sleep(2)
    print 'move: '
    move (1.0, 17.0)
    time.sleep(2) 
    if (ACDC == True):
        return

    rotate(30 , 180, 1)

    time.sleep(2)
    print 'move: '
    move (1.0, 17.0)
    time.sleep(2) 
    if (ACDC == True):
        return

    pen(255,255,255,1,1) #off
    ################################
    rotate(30 , 90, 0) # left
    if (ACDC == True):
        return

    time.sleep(2)
    print 'move: '
    move (1.0, 9.0)
    time.sleep(2)
    if (ACDC == True):
        return 

    pen(255,255,255,1,0) # on

    time.sleep(2)
    print 'move: '
    move (1.0, 10.0)
    time.sleep(2)
    if (ACDC == True):
        return 
    
    rotate(30 , 90, 0)

    time.sleep(2)
    print 'move: '
    move (1.0, 5.0)
    time.sleep(2) 
    if (ACDC == True):
        return

    rotate(30 , 90, 0)

    time.sleep(2)
    print 'move: '
    move (1.0, 5.0)
    time.sleep(2)
    if (ACDC == True):
        return
    
    rotate(30 , 90, 1)

    time.sleep(2)
    print 'move: '
    move (1.0, 5.0)
    time.sleep(2)
    if (ACDC == True):
        return

    rotate(30 , 90, 1)

    time.sleep(2)
    print 'move: '
    move (1.0, 8.0)
    time.sleep(2) 
    if (ACDC == True):
        return

    pen(255,255,255,1,1) # off

    time.sleep(2)
    print 'move: '
    move (1.0, 3.0)
    time.sleep(2)
    if (ACDC == True):
        return
    
    pen(255,255,255,1,0) # off
    
    rotate(30 , 90, 1)

    time.sleep(2)
    print 'move: '
    move (1.0, 18.0)
    time.sleep(2) 
    if (ACDC == True):
        return



if __name__ == '__main__':
    global tolerance, distance, ACDC
    print("Input the tolerance")
    tolerance = input()
    rospy.wait_for_service('reset')
    reset_simulator = rospy.ServiceProxy('reset', EmptyServiceCall)
    reset_simulator()

    try:
        global loop_rate
        rospy.init_node('turtlesim_USI')
        
        loop_rate = rospy.Rate(30)
        #declare velocity publisher
        cmd_vel_topic='/turtle1/cmd_vel'
        velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        
        position_topic = "/turtle1/pose"
        pose_subscriber = rospy.Subscriber(position_topic, Pose, poseCallback)
        
        ########################################################################
        try:
            rospy.wait_for_service('spawn')
            spawner = rospy.ServiceProxy('spawn', Spawn)
            spawner(2, 2, 0, 'turtle2')
        except: 
            pass
        pose_subscriber_2 = rospy.Subscriber("/turtle2/pose", Pose, turtle2Pose)
        ########################################################################

        while not rospy.is_shutdown():
            if (ACDC == False):
               usi_marker()
            else: 
                tutorialMethodPlus()

        


        # teleport = rospy.ServiceProxy('teleport_relative', TeleportRelative)
        # teleport(11.0, 2.0)
        #print 'start reset: '
        #rospy.wait_for_service('reset')
        #reset_turtle = rospy.ServiceProxy('reset', Empty)
        #reset_turtle()
        #print 'end reset: '
        rospy.spin()
       

    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")