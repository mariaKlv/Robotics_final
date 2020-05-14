#!/usr/bin/env python
import rospy
import roslib
import sys
import std_srvs.srv
import sensor_msgs.msg
import time
import numpy as np
import asebaros_msgs.msg
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from geometry_msgs.msg import Pose, Twist, Vector3
from std_srvs.srv import Empty as EmptyServiceCall
from math import pow, atan2, sqrt

tolerance_for_wall = 0.1 # measured in meters

class ThymioControl:
	def __init__ (self):
		rospy.init_node('thymioRover')
		self.rate = rospy.Rate(10)

		#publishers
		self.vel_publisher = rospy.Publisher('/thymio10/cmd_vel', Twist, queue_size = 10)
		self.vel_message = Twist()

		#subscriptions
		self.odom_suscriber = rospy.Subscriber('/thymio10/odom', Odometry, self.update_odom)
		self.center_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/center', Range, self.update_Csensors)
		self.right_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/center_right', Range, self.update_Rsensors)
		self.left_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/center_left', Range, self.update_Lsensors)
		self.left_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/rear_left', Range, self.update_rearLsensors)
		self.left_sensor_suscriber = rospy.Subscriber('/thymio10/proximity/rear_right', Range, self.update_rearRsensors)


		#self.odometry = Odometry()
		self.pose_position = Odometry().pose.pose.position
		self.pose_orientation = Odometry().pose.pose.orientation
		self.dt = 0.0

		#sensors
		self.center_sensor = Range()
		self.right_sensor = Range()
		self.left_sensor = Range()
		self.rearL_sensor = Range()
		self.rearR_sensor = Range()

		#flags and variables
		# TO BE INSERTED
		self.FOUND_WALL = False	
		self.KEEP_VELOCITY = False



	def update_Csensors(self,sen):
		self.center_sensor = round(sen.range,4)
		#print("\ncenter: " + str(self.center_sensor))
	def update_Rsensors(self,sen):
		self.right_sensor = round(sen.range,4)
		#print("right: " + str(self.right_sensor))
	def update_Lsensors(self,sen):
		self.left_sensor = round(sen.range,4)
		#print("left: " + str(self.left_sensor))
	def update_rearLsensors(self,sen):
		self.rearL_sensor = round(sen.range,4)
		#
	def update_rearRsensors(self,sen):
		self.rearR_sensor = round(sen.range,4)
		#

	def update_odom(self, od):
		#odometry = od
		self.pose_position = od.pose.pose.position
		self.pose_orientation = od.pose.pose.orientation

		self.dt += 0.1	#update state

