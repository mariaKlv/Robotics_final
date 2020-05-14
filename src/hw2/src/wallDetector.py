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

tolerance_for_wall = 0.1 # measured in meters

class MyTController:
	def __init__(self):
		# Initialize the node
		rospy.init_node('Mighty_Thymio_Controller') # name of the node

		# Publish to the topic '/thymio10/cmd_vel'
		self.cmd_vel_topic='/thymio10/cmd_vel'
		self.velocity_publisher = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

		# initialize linear and angular velocities to 0
		self.velocity = Twist()
		
		# Odometry topic
		self.odometry_topic = '/thymio10/odom'
		# create pose subscriber
		self.pose_subscriber = rospy.Subscriber(
			self.odometry_topic,  # name of the topic
			Odometry,  # message type
			self.log_odometry  # function that hanldes incoming messages
		)
		self.center_link_subscriber = rospy.Subscriber('thymio10/proximity/center', Range, self.log_center_link)
		self.left_link_subscriber = rospy.Subscriber('thymio10/proximity/left', Range, self.log_left_link)
		self.right_link_subscriber = rospy.Subscriber('thymio10/proximity/right', Range, self.log_right_link)
		

		odom = Odometry()

		self.odom_position = odom.pose.pose.position
		# the rotation, quaternion created from yaw
		self.odom_quat = odom.pose.pose.orientation

		self.center_link = Range()
		self.left_link = Range()
		self.right_link = Range()

		# self.time = rospy.Duration(2)	

		# BOOLEANS 
		self.found_wall = False	
		self.keep_velocity_unchanged = False
		self.perpendicular_to_wall = False	

		# Publish at this rate
		self.rate = rospy.Rate(10)


	def log_odometry(self, data):
		"""Updates robot pose and velocities, and logs pose to console."""

		# Odometry knows how far the robot has gone and how much the angle of rotation is 
		# calculates the position and orientation for us
		self.odom_position = data.pose.pose.position

		# odom_trans.transform.rotation = odom_quat
		# which is the rotation, quaternion created from yaw
		self.odom_quat = data.pose.pose.orientation

	# CENTER CENSOR
	def log_center_link(self, censor_data):
		# it needs somehow to get informed by the input data 
		self.center_link = round(censor_data.range,4)

	# LEFT CENSOR 
	def log_left_link(self, censor_data):
		# it needs somehow to get informed by the input data 
		self.left_link = round(censor_data.range,4)

	# RIGHT CENSOR
	def log_right_link(self, censor_data):
		# it needs somehow to get informed by the input data 
		self.right_link = round(censor_data.range,4)

	def bang_bang_controller(self):	
		""" We are going to implement a bang bang controller that when 
		the tolerance_for_wall has not been reached yet we move forward, 
		otherwise we have to stop instantly and publish our linear velocity """
		if (self.center_link <= tolerance_for_wall or self.left_link <= tolerance_for_wall or self.right_link <= tolerance_for_wall):
			# we should set it somehow less than the value of tolerance in order not to move to fast
			found_wall = True
			self.velocity.linear.x = 0
		else:
			# we have to stop as soon as possible
			# found_wall = True
			self.velocity.linear.x = 0.2
			self.keep_velocity_unchanged = True
		self.velocity_publisher.publish(self.velocity)
		self.velocity.angular.z = 0


	def run(self):
		"""Controls the Thymio."""

		while not rospy.is_shutdown():

			# decide control action
			velocity = self.bang_bang_controller()
			
			# sleep until next step
			self.rate.sleep()

	def check_for_orthogonality(self):
		if (abs(tolerance_for_wall-self.center_link) <= abs(tolerance_for_wall - self.left_link) and \
			abs(tolerance_for_wall-self.center_link) <= abs(tolerance_for_wall - self.right_link)):
			self.perpendicular_to_wall = True
			self.velocity.angular.z = 0 

		else:
			self.perpendicular_to_wall = False
			self.velocity.angular.z = 0.2



if __name__ == '__main__':
	controller = MyTController()

	try:	
		controller.run()
		while (perpendicular_to_wall == True):
			controller.check_for_orthogonality()
	except rospy.ROSInterruptException as e:
		pass
	finally:
		rospy.spin()