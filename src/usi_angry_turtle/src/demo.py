#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import time
import rospy
from turtlesim.msg import *
from turtlesim.srv import *
from geometry_msgs.msg import Twist
from std_srvs.srv import *
import random
from time import time
from math import atan2,pi, sqrt, pow
import sys
from std_srvs.srv import Empty
from std_msgs.msg import String
from math import pow, atan2, sqrt
import turtlesim.srv
import random
import time 
import std_srvs.srv
from turtlesim.srv import SetPen
PI = 3.1415926535897
import tf
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
#################################################
import roslib
roslib.load_manifest('usi_angry_turtle')

targetHuntedTurtles = 1 #How many turtles to hunt until the script ends
tolerance = 1 #The distance until we say the hunter finds the hunted.


#Global Variables I need elsewhere, don't change.
lastDistance = 10
timeToFind = 0
huntedTurtles = 0
totalHuntTime = 0

x=0
y=0
z=0
yaw=0

def poseCallback(pose_message):
    global x
    global y, z, yaw
    x= pose_message.x
    y= pose_message.y
    yaw = pose_message.theta

    #print "pose callback"
    #print ('x = {}'.format(pose_message.x)) #new in python 3
    #print ('y = %f' %pose_message.y) #used in python 2
    #print ('yaw = {}'.format(pose_message.theta)) #new in python 3

def huntedPose(pose_message):
    global turtleTargetx, turtleTargety
    turtleTargetx = pose_message.x
    turtleTargety = pose_message.y

#Create a new Turtle to Hunt
def spawnNewTurtle():
    global turtleTargetx, turtleTargety
    turtleTargetx = random.randint(0, 11)
    turtleTargety = random.randint(0, 11)
    spawnTurtle(turtleTargetx,turtleTargety,0,"turtleTarget")


def move(speed, distance):
        #declare a Twist message to send velocity commands
            vel_msg_m = Twist()
            #get current location 
            x0=x
            y0=y
            
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


def getDistance(x1, y1, x2, y2):
    return sqrt(pow((x2-x1),2) + pow((y2-y1),2))

def tutorialMethodPlus():
    global motion, lastDistance
    distance = getDistance(x, y, turtleTargetx, turtleTargety)
    targetTheta = atan2(turtleTargety - y, turtleTargetx - x)
    if (targetTheta<0):
       targetTheta += 2 * pi
    if (distance <= lastDistance):
        motion.linear.x = 1.5*distance
    else:
        motion.linear.x = .1
    lastDistance=distance
    change = 1
    if (targetTheta - yaw <0):
        change = -1
    motion.angular.z = 4 * (targetTheta - yaw) * change
    velocity_publisher.publish(motion)

def finishHunt():
    global motion
    motion.linear.x = 0
    motion.angular.z = 0
    velocity_publisher.publish(motion)
    print "Total Turtles Hunted"
    print huntedTurtles
    print "Total Seconds"
    print totalHuntTime
    print "Average Find Time"
    print totalHuntTime / huntedTurtles
    try:
        killTurtle("turtleTarget")
    except:
        pass
    
    sys.exit()

def resetHunt():
    global timeToFind, lastDistance
    try:
        killTurtle("turtleTarget")
    except:
        pass
    lastDistance = 10
    clearStage()
    spawnNewTurtle()
    timeToFind = rospy.Time.now().to_sec()

def hunt():
    global totalHuntTime, huntedTurtles, x, y, turtleTargetx, turtleTargety
    
    #Set up board for first hunt.
    resetHunt()
    #Main Loop
    while not rospy.is_shutdown():

        #See how far away hunter is from hunted
        distance = getDistance(x, y, turtleTargetx, turtleTargety)
        #If <= tolerance then it is found
        if (distance <= tolerance):
            print "I am done beaches"
            if (huntedTurtles == targetHuntedTurtles):
                finishHunt()
            
        else: #Didn't find the target, time to use one of the methods!

            tutorialMethodPlus()

        #Sleep to our publish rate       
        rate.sleep()
        



if __name__ == '__main__':
    #first declare at init the instantiated variable
    # ################## PEN #################### #
    # pen = SetPen()
    # pen = rospy.ServiceProxy('/turtle1/set_pen', SetPen)
    
    #rospy.wait_for_service('spawn')
    #spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)
    #spawner(4, 2, 0, 'turtle2')


    try:

        global velocity_publisher, rate, motion
        
        rospy.init_node('turtlesim_USI')

        #declare velocity publisher 1
        cmd_vel_topic='/turtle1/cmd_vel'
        velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        
        #position_topic = "/turtle1/pose"
        #pose_subscriber = rospy.Subscriber(position_topic, Pose, poseCallback)


        #######################################################################
        
        rospy.Subscriber("/turtle1/pose", Pose, poseCallback) #Getting the hunter's Pose
        rospy.Subscriber("/turtleTarget/pose", Pose, huntedPose) #Getting the hunted's Pose
        rate = rospy.Rate(30) #The rate of our publishing
        rospy.wait_for_service('spawn')
        rospy.wait_for_service('kill')
        rospy.wait_for_service('clear')
        clearStage = rospy.ServiceProxy('clear', Empty) #Blanks the Stage
        spawnTurtle = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn) #Can spawn a turtle
        killTurtle = rospy.ServiceProxy('kill', turtlesim.srv.Kill) #Delets a turtle
        setPen = rospy.ServiceProxy('/turtle1/set_pen', SetPen) #Sets the pen color of the hunter
        motion = Twist() #The variable we send out to publish

        hunt()
    except rospy.ROSInterruptException:
        pass
        ##################################################################################
        
        
        rotate(30 , 97, 0)  # up            

        time.sleep(2)
        print 'move: '
        move (1.0, 17.0)
        time.sleep(2)

        rotate(30 , 180, 1) 

        time.sleep(2)
        print 'move: '
        move (1.0, 17.0)
        time.sleep(2)

        rotate(30 , 90, 1)

        time.sleep(2)
        print 'move: '
        move (1.0, 5.0)
        time.sleep(2)

        rotate(30 , 90, 1)

        time.sleep(2)
        print 'move: '
        move (1.0, 17.0)
        time.sleep(2) 

        rotate(30 , 180, 1)

        time.sleep(2)
        print 'move: '
        move (1.0, 17.0)
        time.sleep(2) 

        pen(255,255,255,1,1) #off
        ################################
        rotate(30 , 90, 0) # left
        
        time.sleep(2)
        print 'move: '
        move (1.0, 9.0)
        time.sleep(2) 

        pen(255,255,255,1,0) # on

        time.sleep(2)
        print 'move: '
        move (1.0, 10.0)
        time.sleep(2) 
        
        rotate(30 , 90, 0)

        time.sleep(2)
        print 'move: '
        move (1.0, 5.0)
        time.sleep(2) 

        rotate(30 , 90, 0)

        time.sleep(2)
        print 'move: '
        move (1.0, 5.0)
        time.sleep(2)
        
        rotate(30 , 90, 1)

        time.sleep(2)
        print 'move: '
        move (1.0, 5.0)
        time.sleep(2)

        rotate(30 , 90, 1)

        time.sleep(2)
        print 'move: '
        move (1.0, 8.0)
        time.sleep(2)

        pen(255,255,255,1,1) # off

        time.sleep(2)
        print 'move: '
        move (1.0, 3.0)
        time.sleep(2)
        
        pen(255,255,255,1,0) # off
        
        rotate(30 , 90, 1)

        time.sleep(2)
        print 'move: '
        move (1.0, 18.0)
        time.sleep(2)

        # teleport = rospy.ServiceProxy('teleport_relative', TeleportRelative)
        # teleport(11.0, 2.0)
        print 'start reset: '
        rospy.wait_for_service('reset')
        #reset_turtle = rospy.ServiceProxy('reset', Empty)
        #reset_turtle()
        print 'end reset: '
        rospy.spin()
       

    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")